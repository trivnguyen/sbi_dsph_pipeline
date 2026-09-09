"""Web app for amortized dSph density-profile inference on user catalogs.

Upload a kinematic catalog in the fixed format defined by
user_catalog.py (`ra`, `dec`, `vr`, `vr_err`, plus `distance` or `dm`;
optional membership probability and arbitrarily-named boolean flag
columns, all auto-detected and re-assignable per column in the UI),
fill in the system metadata (half-light radius is required - the model
conditions on it), and get an interactive profile explorer, the
posterior corner plot, and the raw posterior samples back.

Every model the server offers (a Lightning .ckpt plus the
config_snapshot.json written by npe/train_npe.py) is loaded once at
startup and held in memory; the UI picks between them per run, the
same way it picks a catalog. Checkpoints are ~11 MB, so holding all of
them costs nothing. Runs are serialized behind a lock since they share
one device.

A model's radius-unit convention is *not* recorded in its checkpoint or
its training config snapshot, and cannot be inferred from either, so it
is never a free-standing runtime choice: it travels with the model it
belongs to, declared once in npe_inference/configs/models.py (repo
mode) or in models/<name>/model_spec.json (a bundle, written by
package.py).
--radius-units exists only for a checkpoint given by path, which no
registry describes.

Run - in the repo, every registered model whose checkpoint is present:
    python app.py [--model 8p_priorA --model 8p_v3] [--port 8799]

Run - one checkpoint by path, or the packaged bundle's ./model:
    python app.py --model-dir /path/to/model [--radius-units kpc]

then open http://<server>:8799 (or SSH-tunnel the port). See README.md.
"""

import argparse
import base64
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Optional

# Cap math-library thread pools BEFORE numpy/torch load: CPU torch
# defaults to one thread per core, which multiplies across the web
# server's threads and the Jeans worker processes and can exhaust the
# host's thread limit (observed as "libgomp: Thread creation failed"
# on a busy 192-core node). Export a value to override.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '8')

import matplotlib
matplotlib.use('Agg')  # headless server - must precede pyplot import

import numpy as np
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
import app_paths
import inference
import user_catalog

from dsph_analysis import kinematic_io, vdisp, vkurtosis
from tsnpe import prior

app = FastAPI(title='dSph posterior explorer')
app.mount('/static', StaticFiles(directory=_APP_DIR / 'static'),
          name='static')

# Filled by main() before uvicorn starts: models (name -> a loaded
# entry, see _load_entry), default_model, device, output_dir,
# n_workers.
STATE = {}

# One inference at a time (shared device); concurrent requests queue.
_RUN_LOCK = threading.Lock()

# Uploads are files on disk, so keep few. Results are one posterior
# array each (~64 kB) and are kept far longer: the frontend reuses a
# run whose inputs have not changed rather than recomputing it, so a
# job_id can stay referenced - by its CSV link and its corner
# re-render - across many later runs of the *other* dataset. Evicting
# on the upload's schedule would break those links while the run is
# still on screen.
_MAX_UPLOADS = 8
_MAX_RESULTS = 64
_UPLOADS = OrderedDict()  # upload_id -> saved file path
_RESULTS = OrderedDict()  # job_id -> {label, posterior, model}


def _default_workers() -> int:
    """Worker count for the Jeans profile pool: the CPUs actually
    granted to this process where that's knowable (Linux/cgroup),
    capped so a big allocation isn't assumed to be exclusive.
    """
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, min(n, 32))


def _remember(cache: OrderedDict, key: str, value, limit: int) -> None:
    """Insert into a bounded cache, evicting the oldest entry.

    Args:
        cache: The cache to insert into.
        key: Cache key.
        value: Value to store.
        limit: Maximum entries to keep.
    """
    cache[key] = value
    while len(cache) > limit:
        cache.popitem(last=False)


def _b64_png(path: Path) -> str:
    with open(path, 'rb') as f:
        return ('data:image/png;base64,'
                + base64.b64encode(f.read()).decode('ascii'))


def _opt_float(value) -> float:
    """Parse an optional numeric form value ('' / None -> None)."""
    if value is None or value == '':
        return None
    return float(value)


def _summary(
    label: str, info: dict, posterior_kpc: np.ndarray, entry: dict,
) -> dict:
    """JSON-friendly run summary: selection info plus 16/50/84
    posterior percentiles per parameter.

    Args:
        label: Display name for the run.
        info: Selection info from user_catalog.build_target.
        posterior_kpc: Draws with every radius column in log10(r/kpc),
            i.e. inference.to_physical_kpc output. `unit` marks which
            columns those are, so the UI can say so.
        entry: The model entry this run used. Reported back because
            results stay on screen while the model selector moves, so
            the table has to say which model it is showing.
    """
    q = np.percentile(posterior_kpc, [16, 50, 84], axis=0)
    params = [
        dict(name=name, median=float(q[1, i]),
             minus=float(q[1, i] - q[0, i]),
             plus=float(q[2, i] - q[1, i]),
             unit=('log10(r/kpc)'
                   if i in inference.RADIUS_PARAM_INDICES
                   or i == inference.I_COND else ''))
        for i, name in enumerate(prior.ALL_PARAM_NAMES)
    ]
    return dict(label=label, info=info, params=params,
                model=entry['name'],
                model_radius_units=entry['radius_units'],
                n_posterior_samples=int(len(posterior_kpc)))


@app.get('/')
def index():
    return FileResponse(_APP_DIR / 'static' / 'index.html')


def _entry_public(entry: dict) -> dict:
    """The part of a model entry the frontend may see (no tensors)."""
    return dict(
        name=entry['name'], radius_units=entry['radius_units'],
        note=entry['note'],
        model_dir=str(entry['model_dir'] / entry['checkpoint']))


@app.get('/api/config')
def get_config():
    """Server-side facts the UI shows at load time, including every
    model it may choose between.
    """
    return dict(
        device=str(STATE['device']),
        default_model=STATE['default_model'],
        models=[_entry_public(e) for e in STATE['models'].values()],
    )


@app.get('/api/models')
def list_models():
    """The loaded models, for the run-time model selector.

    `radius_units` is reported per model and is not separately
    selectable: it describes how a given flow's radius outputs are
    scaled, so pairing it with a different checkpoint would silently
    shift dm_log_rdm by a factor of r_star. Output is always kpc
    whichever model runs - see inference.to_physical_kpc.
    """
    return dict(
        default=STATE['default_model'],
        models=[_entry_public(e) for e in STATE['models'].values()])


@app.get('/api/systems')
def list_systems():
    """Known system keys from the local_volume_database snapshot, for
    the metadata-prefill dropdown.
    """
    table = kinematic_io.load_meta_table()
    return dict(keys=sorted(table['key'].astype(str)))


@app.get('/api/meta/{key}')
def get_meta(key: str):
    """Prefill values for one known system (fills the metadata form;
    the run itself always uses the form's values).
    """
    try:
        meta = kinematic_io.load_meta(key)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    def _val(quantity):
        v = float(quantity.value)
        return None if np.isnan(v) else v

    return dict(
        key=key,
        center_ra_deg=_val(meta.ra),
        center_dec_deg=_val(meta.dec),
        distance_kpc=_val(meta.distance),
        pmra_masyr=_val(meta.pmra),
        pmdec_masyr=_val(meta.pmdec),
        vlos_systemic_kms=_val(meta.vlos_systemic),
        rhalf_kpc=_val(meta.rhalf_kpc),
        rhalf_kpc_em=_val(meta.rhalf_kpc_em),
        rhalf_kpc_ep=_val(meta.rhalf_kpc_ep),
    )


_UPLOAD_ID_RE = re.compile(r'^[0-9a-f]{32}$')


def resolve_upload(upload_id) -> Optional[Path]:
    """The saved file for `upload_id`, or None if it is gone.

    Falls back to the file on disk when the in-memory entry has been
    evicted, or when the process restarted against a persistent
    --output-dir. The id is checked against the uuid4 hex shape first:
    it reaches the glob below, and a client-supplied path fragment has
    no business there.

    Args:
        upload_id: The id handed out by /api/inspect.

    Returns:
        The catalog's path, or None.
    """
    if not isinstance(upload_id, str) or not _UPLOAD_ID_RE.match(upload_id):
        return None
    known = _UPLOADS.get(upload_id)
    if known is not None and Path(known).exists():
        return known
    for path in sorted(STATE['output_dir'].glob(f'upload_{upload_id}.*')):
        _remember(_UPLOADS, upload_id, path, _MAX_UPLOADS)
        return path
    return None


def _inspect_catalog_file(path: Path) -> dict:
    """Parse a saved catalog and describe its columns.

    Args:
        path: The saved catalog file.

    Returns:
        The inspection payload, plus a best-effort `suggestion`.

    Raises:
        HTTPException: If the file cannot be parsed.
    """
    try:
        df = user_catalog.read_catalog(str(path))
        inspection = user_catalog.inspect_catalog(df)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f'Could not parse catalog: {e}')

    # Best-effort prefill suggestion; never fail the upload over it.
    try:
        suggestion = user_catalog.suggest_system(
            df, inspection['mapping'], kinematic_io.load_meta_table())
    except Exception:
        suggestion = dict(suggested_key=None, reason=None,
                          sep_arcmin=None, candidates=[])
    return dict(suggestion=suggestion, **inspection)


@app.post('/api/inspect')
def inspect(file: UploadFile):
    """Save an uploaded catalog and auto-detect its columns.

    Returns an `upload_id` that /api/run references, so the (possibly
    large) file is only uploaded once.
    """
    upload_id = uuid.uuid4().hex
    suffix = Path(file.filename or 'catalog.csv').suffix or '.csv'
    path = STATE['output_dir'] / f'upload_{upload_id}{suffix}'
    with open(path, 'wb') as f:
        f.write(file.file.read())
    try:
        described = _inspect_catalog_file(path)
    except HTTPException:
        path.unlink(missing_ok=True)
        raise

    _remember(_UPLOADS, upload_id, path, _MAX_UPLOADS)
    return dict(upload_id=upload_id, filename=file.filename,
                **described)


@app.get('/api/upload/{upload_id}')
def get_upload(upload_id: str):
    """Re-describe a catalog this server still holds.

    Lets a loaded settings file pick its catalog back up without the
    user re-uploading it. Returns exactly what /api/inspect returns,
    minus the original file name, which is not kept on disk - the
    settings file carries that.

    Args:
        upload_id: The id recorded in the settings file.

    Returns:
        The inspection payload.

    Raises:
        HTTPException: 404 if the server no longer has the file.
    """
    path = resolve_upload(upload_id)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail='This server no longer has that catalog - it was '
                   'uploaded to a different server, or the run '
                   'directory has been cleared.')
    return dict(upload_id=upload_id, filename=None,
                **_inspect_catalog_file(path))


@app.post('/api/preview')
def preview(payload: dict):
    """Per-star arrays for the manual-selection plots.

    Applies the same flag/membership/query row cuts the run will (see
    user_catalog.select_for_preview), and returns each surviving star's
    catalog row index plus the available plot columns (RA/Dec, and
    pmra/pmdec, vr/[Fe/H] where present). The frontend plots these, lets
    the user exclude/keep stars interactively, and sends the kept row
    indices back to /api/run as `manual_ids`.
    """
    catalog_path = resolve_upload(payload.get('upload_id'))
    if catalog_path is None:
        raise HTTPException(
            status_code=400,
            detail='Unknown upload_id - (re-)upload the catalog first.')
    try:
        df = user_catalog.read_catalog(str(catalog_path))
        mapping = user_catalog.resolve_mapping(
            df, payload.get('columns') or {})
        data = user_catalog.select_for_preview(
            df, mapping,
            flag_requirements=payload.get('flags') or {},
            mem_prob_min=_opt_float(payload.get('mem_prob_min')),
            query=payload.get('query'))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    columns = {}
    for role in user_catalog.PREVIEW_ROLES:
        column = mapping.get(role)
        if column is None or column not in data.columns:
            continue
        values = pd.to_numeric(data[column], errors='coerce').to_numpy()
        if not np.isfinite(values).any():
            continue
        columns[role] = [None if not np.isfinite(v) else float(v)
                         for v in values]
    return dict(
        n=int(len(data)),
        ids=[int(i) for i in data.index.to_numpy()],
        columns=columns,
        pairs=[list(p) for p in user_catalog.PREVIEW_PAIRS
               if p[0] in columns and p[1] in columns],
    )


@app.post('/api/run')
def run(payload: dict):
    """Full inference on a previously-inspected upload: build the
    target, sample the posterior, compute the Jeans/binned/Wolf
    profiles, and return the corner PNG + the interactive-profile JSON.
    """
    catalog_path = resolve_upload(payload.get('upload_id'))
    if catalog_path is None:
        raise HTTPException(
            status_code=400,
            detail='Unknown upload_id - (re-)upload the catalog first.')
    if payload.get('rhalf_kpc') in (None, ''):
        raise HTTPException(
            status_code=400,
            detail='rhalf_kpc is required - the model conditions on it.')
    # The conditioning prior is a window n_sigma wide around the
    # measured r_half, so a zero uncertainty makes it a single point
    # and prior.in_prior_box cuts essentially every draw. The failure
    # surfaces deep inside sample_posterior as "all posterior samples
    # fell outside the prior box", which reads like a model problem,
    # so catch it here where the cause is still visible.
    if (_opt_float(payload.get('rhalf_kpc_em')) or 0.0) <= 0 and \
            (_opt_float(payload.get('rhalf_kpc_ep')) or 0.0) <= 0:
        raise HTTPException(
            status_code=400,
            detail='R_half needs a non-zero uncertainty: the model '
                   'conditions on a window around it, and a window of '
                   'zero width contains no posterior draws. Give at '
                   'least one of the -err / +err values (they are '
                   'prefilled with the published values for a known '
                   'system).')
    entry = _select_model(payload.get('model'))

    label = str(payload.get('label') or 'user_catalog').strip()
    try:
        with _RUN_LOCK:
            # Timed from inside the lock, so a queued request reports its
            # own compute time rather than time spent waiting.
            t_run = time.perf_counter()
            df = user_catalog.read_catalog(str(catalog_path))
            target, info = user_catalog.build_target(
                df, label=label,
                rhalf_kpc=float(payload['rhalf_kpc']),
                rhalf_kpc_em=_opt_float(payload.get('rhalf_kpc_em'))
                or 0.0,
                rhalf_kpc_ep=_opt_float(payload.get('rhalf_kpc_ep'))
                or 0.0,
                columns=payload.get('columns') or {},
                center_ra_deg=_opt_float(payload.get('center_ra_deg')),
                center_dec_deg=_opt_float(
                    payload.get('center_dec_deg')),
                distance_kpc=_opt_float(payload.get('distance_kpc')),
                vlos_systemic_kms=_opt_float(
                    payload.get('vlos_systemic_kms')),
                pmra_masyr=_opt_float(payload.get('pmra_masyr')),
                pmdec_masyr=_opt_float(payload.get('pmdec_masyr')),
                vlos_abs_max=_opt_float(payload.get('vlos_abs_max')),
                mem_prob_min=_opt_float(payload.get('mem_prob_min')),
                flag_requirements=payload.get('flags') or {},
                apply_perspective_corr=bool(
                    payload.get('apply_perspective_corr', True)),
                query=payload.get('query'),
                radius_min=_opt_float(payload.get('radius_min')),
                radius_max=_opt_float(payload.get('radius_max')),
                radius_unit=payload.get('radius_unit') or 'kpc',
                manual_ids=payload.get('manual_ids'),
            )

            n_samples = int(payload.get('n_samples') or 1000)
            n_bins = int(payload.get('n_bins') or 4)
            t_sample = time.perf_counter()
            # sample_posterior returns draws in the model's own radius
            # units and cuts them against the prior box in that space;
            # everything below this point wants kpc.
            posterior_model_units = inference.sample_posterior(
                entry['model'], entry['prior'], target,
                entry['norm_dict'], entry['pre_transforms_config'],
                n_samples=n_samples, n_mc_conditioning=n_samples,
                conditioning_dist='gaussian', return_log_prob=False,
                batch_size=int(payload.get('batch_size') or 512))
            sampling_sec = time.perf_counter() - t_sample
            posterior = inference.to_physical_kpc(
                posterior_model_units, entry['radius_units'])

            # Optional rejection cut on the inner slope gamma. We keep
            # the requested sample count fixed and just drop the draws
            # outside the range, reporting how many survive. gamma is
            # dimensionless, so the kpc conversion above leaves it
            # alone and the cut means the same thing on either side.
            n_requested = len(posterior)
            gamma_min = _opt_float(payload.get('gamma_min'))
            gamma_max = _opt_float(payload.get('gamma_max'))
            posterior = inference.restrict_gamma(
                posterior, gamma_min, gamma_max)
            if len(posterior) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f'No posterior samples fall in gamma '
                        f'[{gamma_min}, {gamma_max}]. Widen the range '
                        f'or increase the posterior-sample count.'))

            vdisp_profile = vdisp.calc_vdisp_los_binned(
                target.R_proj_kpc, target.vlos_kms,
                target.vlos_err_kms,
                nbins_min=n_bins, nbins_max=n_bins, verbose=False)
            vkurtosis_profile = vkurtosis.calc_kurtosis_los_binned(
                target.R_proj_kpc, target.vlos_kms,
                target.vlos_err_kms,
                nbins_min=n_bins, nbins_max=n_bins, verbose=False)
            jeans = inference.calc_jeans_profiles(
                posterior, inference.R_VEC_KPC,
                n_samples=int(payload.get('n_profile_samples') or 500),
                n_workers=STATE['n_workers'])

            wolf = inference.calc_wolf_mass(
                target.vlos_kms, target.vlos_err_kms, target.rhalf_kpc)
            key = str(payload.get('key') or '').strip()
            wolf['literature'] = (
                inference.load_literature_mass_wolf(key) if key
                else None)

            job_id = uuid.uuid4().hex
            corner_path = STATE['output_dir'] / f'{job_id}_corner.png'
            inference.plot_corner(
                posterior, corner_path,
                _corner_title(dict(label=label, model=entry['name'])))
            _remember(_RESULTS, job_id,
                      dict(label=label, posterior=posterior,
                           model=entry['name']), _MAX_RESULTS)
            profiles = inference.profiles_payload(
                inference.R_VEC_KPC, jeans, vdisp_profile,
                vkurtosis_profile, wolf)
            walltime_sec = time.perf_counter() - t_run
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # sample_posterior raises this when every draw falls outside
        # the prior box. With a valid catalog that almost always means
        # the server was launched with the wrong --radius-units, so say
        # so rather than showing a bare traceback.
        raise HTTPException(
            status_code=500,
            detail=f'{e}\n\nModel {entry["name"]} is registered as '
                   f'radius_units={entry["radius_units"]}. If it was '
                   f'in fact trained under the other convention, every '
                   f'draw lands outside the prior box exactly like '
                   f'this - check its entry in '
                   f'npe_inference/configs/models.py.')
    except Exception:
        raise HTTPException(status_code=500,
                            detail=traceback.format_exc())

    summary = _summary(label, info, posterior, entry)
    summary.update(n_posterior_requested=n_requested,
                   gamma_min=gamma_min, gamma_max=gamma_max,
                   walltime_sec=round(walltime_sec, 2),
                   walltime_sampling_sec=round(sampling_sec, 2))
    return dict(
        job_id=job_id,
        summary=summary,
        corner_png=_b64_png(corner_path),
        profiles=profiles,
    )


@app.get('/api/posterior/{job_id}')
def download_posterior(job_id: str):
    """The run's posterior samples as CSV, one column per parameter.

    Radii are log10(r / kpc) - the `_kpc` suffixes are the difference
    between these columns and the model's own output, which holds
    df_log_ra (and, for priorB models, dm_log_rdm) in r_star units.
    """
    if job_id not in _RESULTS:
        raise HTTPException(
            status_code=404,
            detail='Unknown or expired job_id - run inference again.')
    c = _RESULTS[job_id]
    csv = pd.DataFrame(
        c['posterior'],
        columns=inference.KPC_PARAM_NAMES).to_csv(index=False)
    safe = c['label'].replace(' ', '_').replace('/', '_')
    return PlainTextResponse(csv, media_type='text/csv', headers={
        'Content-Disposition':
            f'attachment; filename="{safe}_posterior.csv"'})


@app.get('/api/posteriors')
def download_posteriors(jobs: str):
    """Several runs' posteriors stacked into one tidy CSV.

    The per-run columns are unchanged; two identifier columns are
    prepended so the rows stay separable once concatenated. That is the
    shape the comparison modes produce - several runs over one catalog
    differing only in the model, or one model over several selections -
    so a single `groupby` recovers whichever axis was varied.

    Args:
        jobs: Comma-separated job_ids, in the order to stack them.

    Returns:
        A text/csv attachment.

    Raises:
        HTTPException: If `jobs` is empty or names an unknown run.
    """
    job_ids = [j for j in (jobs or '').split(',') if j.strip()]
    if not job_ids:
        raise HTTPException(
            status_code=400, detail='No job_ids given.')
    missing = [j for j in job_ids if j not in _RESULTS]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f'Unknown or expired job_id(s): {", ".join(missing)} '
                   '- re-run those datasets.')

    frames = []
    for job_id in job_ids:
        c = _RESULTS[job_id]
        frame = pd.DataFrame(
            c['posterior'], columns=inference.KPC_PARAM_NAMES)
        frame.insert(0, 'model', c['model'])
        frame.insert(0, 'label', c['label'])
        frames.append(frame)
    csv = pd.concat(frames, ignore_index=True).to_csv(index=False)
    return PlainTextResponse(csv, media_type='text/csv', headers={
        'Content-Disposition':
            f'attachment; filename="posteriors_{len(job_ids)}runs.csv"'})


@app.post('/api/corner/{job_id}')
def regenerate_corner(job_id: str, options: dict):
    """Re-render the corner PNG for a finished run with new styling
    (smoothing, bin count, contour style, data points) from the stored
    posterior - no re-inference needed.

    The stored samples are already in kpc, so this re-renders exactly
    what the run produced; only the styling changes.
    """
    if job_id not in _RESULTS:
        raise HTTPException(
            status_code=404,
            detail='Unknown or expired job_id - run inference again.')
    c = _RESULTS[job_id]
    opts = dict(
        smooth=_opt_float(options.get('smooth')),
        bins=int(options.get('bins') or 20),
        contours=str(options.get('contours') or 'default'),
        plot_datapoints=bool(options.get('plot_datapoints', True)),
    )
    corner_path = STATE['output_dir'] / f'{job_id}_corner.png'
    try:
        inference.plot_corner(
            c['posterior'], corner_path, _corner_title(c), options=opts)
    except Exception:
        raise HTTPException(status_code=500,
                            detail=traceback.format_exc())
    return dict(corner_png=_b64_png(corner_path))


def _corner_title(cached: dict) -> str:
    """Corner-plot title for a cached run: the label plus the model
    that produced it, since a server offers several.
    """
    return f'{cached["label"]} / {cached["model"]}'


def _select_model(name) -> dict:
    """The loaded model entry a request asked for.

    Args:
        name: Model name from the request payload, or None/'' to take
            the server's default.

    Returns:
        The entry, as built by _load_entry.

    Raises:
        HTTPException: If `name` is not one this server loaded.
    """
    key = str(name or '').strip() or STATE['default_model']
    if key not in STATE['models']:
        raise HTTPException(
            status_code=400,
            detail=f'Unknown model {key!r}. This server loaded: '
                   f'{", ".join(STATE["models"])}.')
    return STATE['models'][key]


def _resolve_checkpoint_filename(model_dir: Path, filename: str) -> str:
    """Use `filename` if it exists; otherwise fall back to the single
    .ckpt file in model_dir (so a bundle just drops any checkpoint in).
    """
    if (model_dir / filename).exists():
        return filename
    candidates = sorted(model_dir.glob('*.ckpt'))
    if len(candidates) == 1:
        return candidates[0].name
    raise FileNotFoundError(
        f'No {filename} in {model_dir} and {len(candidates)} .ckpt '
        'candidates - pass --checkpoint-filename explicitly.')


MODEL_SPEC_FILENAME = 'model_spec.json'


def _load_model_spec(model_dir: Path) -> dict:
    """Read `model_spec.json` from a model directory, if it has one.

    package.py writes this file into the bundle so a bundle user never
    has to know the model's radius convention. A model directory
    pointed at directly in the repo normally has no such file, and the
    setting comes from --radius-units instead.

    Args:
        model_dir: Directory holding the checkpoint.

    Returns:
        The parsed spec, or an empty dict if the file is absent.

    Raises:
        ValueError: If the file exists but is not valid JSON.
    """
    path = model_dir / MODEL_SPEC_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f'{path} is not valid JSON: {e}')


def _resolve_radius_units(cli_value, spec: dict, model_dir: Path) -> str:
    """Settle a by-path model's radius convention, or fail loudly.

    There is deliberately no default. The convention is not recorded in
    the checkpoint or in the training config snapshot, and guessing it
    from a directory name would silently shift `dm_log_rdm` by a factor
    of r_star - numbers that look entirely plausible. npe_inference's
    configs/models.py takes the same position: registering a model is a
    deliberate act, and an unregistered one is an error.

    Only reached for --model-dir. Registered models carry the value in
    that same configs/models.py, and a bundle carries it in its
    model_spec.json, so neither path asks the user anything.

    Args:
        cli_value: --radius-units, or None.
        spec: Parsed model_spec.json (may be empty).
        model_dir: Model directory, for the error message.

    Returns:
        'kpc' or 'rstar'.

    Raises:
        SystemExit: If neither source supplies a value.
        ValueError: If the value is not a recognized convention.
    """
    units = cli_value or spec.get('radius_units')
    if units is None:
        raise SystemExit(
            f'No radius convention for {model_dir}: it has no '
            f'{MODEL_SPEC_FILENAME} and --radius-units was not given.\n'
            'A checkpoint does not record which convention it was '
            'trained under, and guessing wrong silently shifts '
            'dm_log_rdm by a factor of r_star, so this is not '
            'defaulted. Either drop --model-dir and name a registered '
            'model with --model, or say which this one is. Registered '
            'models (npe_inference/configs/models.py):\n'
            '  8p_priorA  8p_ZhaoPlumCOM        --radius-units kpc\n'
            '  8p_v3      8p_ZhaoPlumCOM_v3     --radius-units rstar\n'
            '  8p_v3_5M   8p_ZhaoPlumCOM_v3_5M  --radius-units rstar')
    if units not in prior.RADIUS_UNITS_CHOICES:
        raise ValueError(
            f'radius_units={units!r} not recognized; must be one of '
            f'{prior.RADIUS_UNITS_CHOICES}')
    return units


def _load_entry(spec: dict, device) -> dict:
    """Load one model into a ready-to-run STATE['models'] entry.

    Args:
        spec: `{name, model_dir, checkpoint, radius_units, note}`, from
            app_paths.load_registry or built by _build_specs.
        device: torch device to put the flow on.

    Returns:
        `spec` plus the loaded `model`, its `norm_dict` and
        `pre_transforms_config`, and the `prior` whose box
        `sample_posterior` cuts draws against - which is why the prior
        is per-model rather than global: its radius columns are
        interpreted in that model's own units.
    """
    print(f'[Model] {spec["name"]}: loading '
          f'{spec["model_dir"] / spec["checkpoint"]} on {device}...')
    model, norm_dict, pre_transforms_config = inference.load_model(
        str(spec['model_dir']), spec['checkpoint'], device)
    entry = dict(spec)
    entry.update(
        model=model, norm_dict=norm_dict,
        pre_transforms_config=pre_transforms_config,
        prior=inference.build_prior(
            spec['radius_units'], prior_min=spec.get('prior_min'),
            prior_max=spec.get('prior_max')))
    print(f'[Model] {spec["name"]}: {entry["prior"]}')
    if spec.get('note'):
        print(f'[Model] {spec["name"]}: {spec["note"]}')
    return entry


def _spec_from_dir(model_dir: Path, args) -> dict:
    """One unloaded spec for a checkpoint directory given by path.

    Args:
        model_dir: Directory holding the .ckpt, its
            config_snapshot.json, and optionally a model_spec.json.
        args: Parsed command line, for --checkpoint-filename and the
            --radius-units fallback.

    Returns:
        `{name, model_dir, checkpoint, radius_units, note, ...}`.
    """
    spec = _load_model_spec(model_dir)
    return dict(
        spec,
        name=spec.get('name') or model_dir.name,
        model_dir=model_dir,
        checkpoint=_resolve_checkpoint_filename(
            model_dir, args.checkpoint_filename),
        radius_units=_resolve_radius_units(
            args.radius_units, spec, model_dir),
        note=spec.get('note', ''))


def _build_specs(args) -> list[dict]:
    """Decide which models this server will offer, before loading any.

    Four mutually exclusive sources, most explicit first:

    1. `--model-dir`: exactly that checkpoint. Nothing describes it, so
       its convention comes from a `model_spec.json` beside it or from
       `--radius-units`.
    2. A bundle's `./models/<name>/`, one subdirectory per model, each
       with its own `model_spec.json`. This is what package.py writes.
    3. A bundle's `./model`, the single-model layout package.py wrote
       before `models/` existed. Kept so an old bundle still runs.
    4. The registry (`app_paths.load_registry`), optionally narrowed to
       the names given by `--model`. The repo default.

    Args:
        args: Parsed command line.

    Returns:
        Unloaded specs in menu order, the first being the default.

    Raises:
        SystemExit: If no model can be found, or if `--model` names one
            that is not registered.
    """
    if args.model_dir is not None:
        return [_spec_from_dir(Path(args.model_dir), args)]

    bundled = sorted(d for d in (_APP_DIR / 'models').glob('*')
                     if d.is_dir())
    if bundled:
        return [_spec_from_dir(d, args) for d in bundled]
    if (_APP_DIR / 'model').is_dir():
        return [_spec_from_dir(_APP_DIR / 'model', args)]

    registry = app_paths.load_registry()
    if not registry:
        raise SystemExit(
            'No models to serve: no ./models or ./model directory next '
            'to app.py, no --model-dir, and no usable registry at '
            f'{app_paths.NPE_INFERENCE_DIR}/configs/models.py with '
            f'checkpoints under {app_paths.MODEL_WORKDIR}. Point '
            'NPE_INFERENCE_DIR / NPE_MODEL_WORKDIR at them, or pass '
            '--model-dir with --radius-units.')
    if args.model:
        unknown = [m for m in args.model if m not in registry]
        if unknown:
            raise SystemExit(
                f'--model {", ".join(unknown)}: not registered (or the '
                f'checkpoint is missing from {app_paths.MODEL_WORKDIR}). '
                f'Available: {", ".join(registry)}.')
        return [registry[m] for m in args.model]
    return list(registry.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--model', action='append', default=[], metavar='NAME',
        help='A registered model to offer, by its name in '
             'npe_inference/configs/models.py (8p_priorA, 8p_v3, '
             '8p_v3_5M). Repeatable; the first is the default '
             'selection. Omit to offer every registered model whose '
             'checkpoint is present.')
    parser.add_argument(
        '--model-dir', default=None,
        help='Serve exactly one checkpoint, by path to the directory '
             'holding the .ckpt and its config_snapshot.json. Bypasses '
             'the registry; needs --radius-units unless the directory '
             f'has a {MODEL_SPEC_FILENAME}. Omit it in a bundle, which '
             'is served from its own ./models directory.')
    parser.add_argument('--checkpoint-filename', default='model.ckpt')
    parser.add_argument(
        '--radius-units', choices=list(prior.RADIUS_UNITS_CHOICES),
        default=None,
        help="The radius convention of the --model-dir checkpoint: "
             "'kpc' for priorA (8p_ZhaoPlumCOM) or 'rstar' for priorB "
             '(8p_ZhaoPlumCOM_v3). It cannot be read off a checkpoint. '
             'Ignored for --model, which takes each model\'s declared '
             'convention from the registry.')
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8799)
    parser.add_argument(
        '--profile-workers', type=int, default=_default_workers(),
        help='Process count for the Jeans-profile pool (default: '
             'available CPUs, capped at 32).')
    parser.add_argument(
        '--output-dir', default=None,
        help='Where uploads and rendered PNGs go (default: a fresh '
             'temporary directory).')
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        import tempfile
        output_dir = Path(tempfile.mkdtemp(prefix='dsph_webapp_'))

    specs = _build_specs(args)
    device = torch.device(args.device)
    models = OrderedDict(
        (spec['name'], _load_entry(spec, device)) for spec in specs)

    STATE.update(
        models=models, default_model=next(iter(models)), device=device,
        output_dir=output_dir, n_workers=max(1, args.profile_workers))
    print(f'[Server] Models: {", ".join(models)} '
          f'(default {STATE["default_model"]})')
    print(f'[Server] Output dir: {output_dir}')
    print(f'[Server] http://{args.host}:{args.port}')
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

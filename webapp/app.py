"""Web app for amortized dSph density-profile inference on user catalogs.

Upload a kinematic catalog in the fixed format defined by
user_catalog.py (`ra`, `dec`, `vr`, `vr_err`, plus `distance` or `dm`;
optional membership probability and arbitrarily-named boolean flag
columns, all auto-detected and re-assignable per column in the UI),
fill in the system metadata (half-light radius is required - the model
conditions on it), and get an interactive profile explorer, the
posterior corner plot, and the raw posterior samples back.

One pretrained NPE model directory (a Lightning .ckpt plus the
config_snapshot.json written by npe/train_npe.py) is loaded once at
startup and shared by every request; runs are serialized behind a lock
since they share one device.

Run (the packaged bundle defaults to its own ./model directory):
    python app.py [--model-dir /path/to/model] [--port 8799]

then open http://<server>:8799 (or SSH-tunnel the port). See README.md.
"""

import argparse
import base64
import os
import sys
import threading
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path

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
import inference
import user_catalog

from dsph_analysis import kinematic_io, vdisp, vkurtosis
from tsnpe import prior

app = FastAPI(title='dSph posterior explorer')
app.mount('/static', StaticFiles(directory=_APP_DIR / 'static'),
          name='static')

# Filled by main() before uvicorn starts: model, norm_dict,
# pre_transforms_config, device, model_dir, output_dir, n_workers.
STATE = {}

# One inference at a time (shared device); concurrent requests queue.
_RUN_LOCK = threading.Lock()

_MAX_CACHED = 8
_UPLOADS = OrderedDict()  # upload_id -> saved file path
_RESULTS = OrderedDict()  # job_id -> {label, posterior}


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


def _remember(cache: OrderedDict, key: str, value) -> None:
    """Insert into a bounded cache, evicting the oldest entry."""
    cache[key] = value
    while len(cache) > _MAX_CACHED:
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


def _summary(label: str, info: dict, posterior: np.ndarray) -> dict:
    """JSON-friendly run summary: selection info plus 16/50/84
    posterior percentiles per parameter.
    """
    q = np.percentile(posterior, [16, 50, 84], axis=0)
    params = [
        dict(name=name, median=float(q[1, i]),
             minus=float(q[1, i] - q[0, i]),
             plus=float(q[2, i] - q[1, i]))
        for i, name in enumerate(prior.ALL_PARAM_NAMES)
    ]
    return dict(label=label, info=info, params=params,
                n_posterior_samples=int(len(posterior)))


@app.get('/')
def index():
    return FileResponse(_APP_DIR / 'static' / 'index.html')


@app.get('/api/config')
def get_config():
    """Static server-side facts the UI shows at load time."""
    return dict(
        model=str(STATE['model_dir']),
        device=str(STATE['device']),
    )


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
        df = user_catalog.read_catalog(str(path))
        inspection = user_catalog.inspect_catalog(df)
    except Exception as e:
        path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400, detail=f'Could not parse catalog: {e}')

    # Best-effort prefill suggestion; never fail the upload over it.
    try:
        suggestion = user_catalog.suggest_system(
            df, inspection['mapping'], kinematic_io.load_meta_table())
    except Exception:
        suggestion = dict(suggested_key=None, reason=None,
                          sep_arcmin=None, candidates=[])

    _remember(_UPLOADS, upload_id, path)
    return dict(upload_id=upload_id, filename=file.filename,
                suggestion=suggestion, **inspection)


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
    upload_id = payload.get('upload_id')
    if upload_id not in _UPLOADS:
        raise HTTPException(
            status_code=400,
            detail='Unknown upload_id - (re-)upload the catalog first.')
    try:
        df = user_catalog.read_catalog(str(_UPLOADS[upload_id]))
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
    upload_id = payload.get('upload_id')
    if upload_id not in _UPLOADS:
        raise HTTPException(
            status_code=400,
            detail='Unknown upload_id - (re-)upload the catalog first.')
    if payload.get('rhalf_kpc') in (None, ''):
        raise HTTPException(
            status_code=400,
            detail='rhalf_kpc is required - the model conditions on it.')

    label = str(payload.get('label') or 'user_catalog').strip()
    try:
        with _RUN_LOCK:
            df = user_catalog.read_catalog(str(_UPLOADS[upload_id]))
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
            posterior = inference.sample_posterior(
                STATE['model'], target, STATE['norm_dict'],
                STATE['pre_transforms_config'],
                n_samples=n_samples, n_mc_conditioning=n_samples,
                conditioning_dist='gaussian', return_log_prob=False,
                batch_size=int(payload.get('batch_size') or 512))

            # Optional rejection cut on the inner slope gamma. We keep
            # the requested sample count fixed and just drop the draws
            # outside the range, reporting how many survive.
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
            inference.plot_corner(posterior, corner_path, label)
            _remember(_RESULTS, job_id,
                      dict(label=label, posterior=posterior))
            profiles = inference.profiles_payload(
                inference.R_VEC_KPC, jeans, vdisp_profile,
                vkurtosis_profile, wolf)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500,
                            detail=traceback.format_exc())

    summary = _summary(label, info, posterior)
    summary.update(n_posterior_requested=n_requested,
                   gamma_min=gamma_min, gamma_max=gamma_max)
    return dict(
        job_id=job_id,
        summary=summary,
        corner_png=_b64_png(corner_path),
        profiles=profiles,
    )


@app.post('/api/corner/{job_id}')
def regenerate_corner(job_id: str, options: dict):
    """Re-render the corner PNG for a finished run with new styling
    (smoothing, bin count, contour style, data points) from the stored
    posterior - no re-inference needed.
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
            c['posterior'], corner_path, c['label'], options=opts)
    except Exception:
        raise HTTPException(status_code=500,
                            detail=traceback.format_exc())
    return dict(corner_png=_b64_png(corner_path))


@app.get('/api/posterior/{job_id}')
def download_posterior(job_id: str):
    """The run's raw posterior samples as CSV (physical units, one
    column per parameter).
    """
    if job_id not in _RESULTS:
        raise HTTPException(
            status_code=404,
            detail='Unknown or expired job_id - run inference again.')
    c = _RESULTS[job_id]
    csv = pd.DataFrame(
        c['posterior'],
        columns=prior.ALL_PARAM_NAMES).to_csv(index=False)
    safe = c['label'].replace(' ', '_').replace('/', '_')
    return PlainTextResponse(csv, media_type='text/csv', headers={
        'Content-Disposition':
            f'attachment; filename="{safe}_posterior.csv"'})


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--model-dir', default=str(_APP_DIR / 'model'),
        help='Directory with the pretrained checkpoint (.ckpt) and its '
             'config_snapshot.json (default: ./model next to app.py).')
    parser.add_argument('--checkpoint-filename', default='model.ckpt')
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

    model_dir = Path(args.model_dir)
    ckpt_name = _resolve_checkpoint_filename(
        model_dir, args.checkpoint_filename)
    device = torch.device(args.device)
    print(f'[Model] Loading {model_dir / ckpt_name} on {device}...')
    model, norm_dict, pre_transforms_config = inference.load_model(
        str(model_dir), ckpt_name, device)
    STATE.update(
        model=model, norm_dict=norm_dict,
        pre_transforms_config=pre_transforms_config, device=device,
        model_dir=model_dir, output_dir=output_dir,
        n_workers=max(1, args.profile_workers))
    print(f'[Server] Output dir: {output_dir}')
    print(f'[Server] http://{args.host}:{args.port}')
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

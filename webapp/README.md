# dSph posterior explorer — web app

Browser-based amortized inference for people other than the model
owner: upload a kinematic catalog, fill in the system metadata, and
get an interactive profile explorer (density / enclosed mass /
anisotropy / LOS dispersion / LOS kurtosis — mouse zoom/pan, hover
readout, PNG export), the posterior corner plot, a Wolf-mass sanity
check, and the raw posterior samples as CSV.

Started as a port of `plotting/posterior_explorer.ipynb`, but is
deliberately self-contained: `inference.py` replaces
`plotting/posterior_diagnostics.py` + `register_run.py` here so that
`package.py` can ship the whole thing as a portable bundle.

It imports the pipeline's `tsnpe` package from `../tsnpe`, which moves
with the checkout, and `dsph_analysis` by absolute path since that one
lives outside the repo. See `app_paths.py`, which is the single
definition of both (and of the bundle's vendored copies, which
deliberately do *not* fall back to the repo). The defaults match this
machine and are overridable by the same environment variables
`npe_inference/npe_infer/paths.py` uses:

```bash
export MY_MODULES_DIR=~/my_modules
```

The list of models it can serve comes from the same place the rest of
the project's does, `npe_inference/configs/models.py`, found via two
more of those variables:

```bash
export NPE_INFERENCE_DIR=~/projects/sbi_dsph/npe_inference
export NPE_MODEL_WORKDIR=/scratch/tvnguyen/trained_models/dsph_npe
```

## Run from the repo

```bash
source ~/.venvs/torch/bin/activate
python app.py --port 8799
```

That loads every registered model whose checkpoint is on this
filesystem and offers them in a dropdown, so you can put one catalog
through all of them without restarting. Narrow the menu, and pick
which is selected by default, with `--model`:

```bash
python app.py --model 8p_v3 --model 8p_priorA --port 8799
```

Then open `http://<server>:8799`, or tunnel:
`ssh -L 8799:localhost:8799 <server>`.

There is no authentication — only expose the port to people you'd let
run jobs on this machine.

### The models, and their radius conventions

`npe_inference/configs/models.py` is the authority; the app reads it
rather than keeping a second copy, because a second list that drifted
from the first would be invisible until the numbers were already
wrong:

| `--model`   | run                    | radii            |
|-------------|------------------------|------------------|
| `8p_priorA` | `8p_ZhaoPlumCOM`       | `kpc`   (priorA) |
| `8p_v3`     | `8p_ZhaoPlumCOM_v3`    | `rstar` (priorB) |
| `8p_v3_5M`  | `8p_ZhaoPlumCOM_v3_5M` | `rstar` (priorB) |

Radius units are a property of a checkpoint, not a setting, so they
are not separately selectable at runtime: they travel with the model
you pick, and the output is converted to kpc either way (see *Output
units*). The convention is not recorded in the checkpoint, nor in the
`config_snapshot.json` that `npe/train_npe.py` writes, and cannot be
inferred from either. Guessing wrong does not raise; it shifts
`dm_log_rdm` by a factor of `r_star` and yields plausible-looking
profiles — hence the one declaration, in one file.

To serve a checkpoint that file does not list, name it by path and
assert its convention yourself:

```bash
python app.py \
    --model-dir /scratch/tvnguyen/trained_models/dsph_npe/8p_ZhaoPlumCOM/sfaqzcwx/checkpoints \
    --checkpoint-filename last.ckpt \
    --radius-units kpc
```

`--model-dir` accepts any directory holding a training checkpoint
(`.ckpt`) with its `config_snapshot.json` alongside or a few parents
up. It bypasses the registry and serves exactly that one model. A
packaged bundle is the same path: each of its `models/<name>/`
directories carries a `model_spec.json`, so no flag is needed.

## Package for deployment elsewhere

```bash
python package.py --model 8p_priorA --model 8p_v3
```

This builds `dist/dsph_explorer/` — the app, the models, private
copies of `tsnpe`, `jgnn` (models + transforms only, no wandb/h5py),
and `dsph_analysis` (with its local_volume_database snapshot), plus
`requirements.txt`, a CPU `Dockerfile`, an example catalog, and its
own README. Copy that directory anywhere (~28 MB for two models,
mostly vendored plotly.js), `pip install -r requirements.txt`, and
`python app.py` — no repo, cluster filesystem, or network needed. It
runs fine on CPU (a few minutes per run instead of ~20 s).

Ship as many models as the recipient should be able to compare: each
lands in `models/<name>/` with its declared convention in a
`model_spec.json`, and the bundled app gets the same dropdown. A
checkpoint is ~11 MB, so the second one is nearly free. For a
checkpoint the registry does not list, `--checkpoint-dir DIR
--radius-units {kpc,rstar}` ships exactly that one.

Note it needs a Python host (a VPS, lab server, etc.) — the model
can't run on a static-only website.

## Data format

One row per star; CSV, ECSV, or FITS table. Required columns
(case-insensitive; common aliases like `RA`, `vlos`, `vrad_err` are
auto-matched — see `COLUMN_ALIASES` in `user_catalog.py` — and every
assignment can be corrected per column in the UI after upload):

| column     | unit | meaning                                   |
|------------|------|--------------------------------------------|
| `ra`       | deg  | right ascension (ICRS)                    |
| `dec`      | deg  | declination (ICRS)                        |
| `vr`       | km/s | heliocentric line-of-sight velocity       |
| `vr_err`   | km/s | its 1-sigma uncertainty                   |
| `distance` | kpc  | per-star distance — **or** `dm` [mag]; the
                      median sets the system distance. If neither is
                      present, enter the distance in the form. |

Optional, auto-detected: a membership-probability column (threshold
cut) and arbitrarily-named boolean flag columns (ignore / require true
/ require false).

## System metadata

**R_half [kpc]** and at least one of its uncertainties are required.
The model conditions on a window around R_half whose width is that
uncertainty, so a zero-width window admits no posterior draws at all;
the app rejects that up front rather than letting it surface as an
empty posterior. Center, systemic velocity, and distance default to
data-driven medians when blank; proper motions are only needed for the
perspective-rotation correction. Systems in the bundled
local_volume_database snapshot can be prefilled by key (which also
enables the literature Wolf-mass marker).

## Output units

Everything the app hands back — the profile panels, the corner plot,
the summary table, and the posterior CSV — is in physical kpc.
`sample_posterior` returns draws in the model's *own* radius units, so
`inference.to_physical_kpc` runs between the two, byte-for-byte the
same conversion as `plotting/posterior_diagnostics.py` and
`npe_infer.sampling`:

- `df_log_ra` is `log10(r_a / r_star)` under **both** conventions —
  the priorA and priorB training simulators both wrote an explicit
  `r_a = 10 ** log_ra * r_star` — so it is always shifted.
- `dm_log_rdm` is shifted only for a model whose radii are in
  `rstar`.

The CSV columns are named `*_kpc` to mark the ones this affects, since
a downloaded file outlives any note about which convention it holds.
On the corner plot, the two radius panels carry a dashed marker at the
median `log10(r_star / kpc)`: both parameters are bounded relative to
`r_star` rather than in absolute kpc, so where a draw sits relative to
it is the part that carries information.

## Notes

- Runs are serialized behind a lock (one shared device); concurrent
  requests queue and simply take longer.
- Axis ranges are mouse-driven now (drag to zoom, double-click to
  reset) — there is no server-side "replot" step.
- The Jeans-profile worker count is a server-side setting
  (`--profile-workers`, default = available CPUs capped at 32), not a
  UI field.
- The last 8 runs/uploads are cached in memory; older `job_id`s expire
  and need a re-run.
- `sample_posterior` cuts its draws to the prior box, so a run returns
  slightly fewer samples than requested (~95% of `n_samples`, in
  practice). If it returns *none*, the error names the two things that
  cause it: a zero-width R_half window, or a model whose declared
  radius convention is wrong.
- Models are loaded once at startup and held, so switching between
  them mid-session costs nothing; runs still queue behind the same
  lock.

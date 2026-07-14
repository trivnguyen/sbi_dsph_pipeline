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

## Run from the repo

```bash
source ~/.venvs/torch/bin/activate
python app.py \
    --model-dir /scratch/tvnguyen/trained_models/npe/8p_ZhaoPlumCOM/sfaqzcwx/checkpoints \
    --checkpoint-filename last.ckpt \
    --port 8799
```

`--model-dir` accepts any directory holding a training checkpoint
(`.ckpt`) with its `config_snapshot.json` alongside or a few parents
up. Then open `http://<server>:8799`, or tunnel:
`ssh -L 8799:localhost:8799 <server>`.

There is no authentication — only expose the port to people you'd let
run jobs on this machine.

## Package for deployment elsewhere

```bash
python package.py \
    --checkpoint-dir /scratch/tvnguyen/trained_models/npe/8p_ZhaoPlumCOM/sfaqzcwx/checkpoints \
    --checkpoint-filename last.ckpt
```

This builds `dist/dsph_explorer/` — the app, the model, private copies
of `tsnpe`, `jgnn` (models + transforms only, no wandb/h5py), and
`dsph_analysis` (with its local_volume_database snapshot), plus
`requirements.txt`, a CPU `Dockerfile`, an example catalog, and its own
README. Copy that directory anywhere (it's ~60 MB, mostly the
checkpoint and vendored plotly.js), `pip install -r requirements.txt`,
and `python app.py` — no repo, cluster filesystem, or network needed.
It runs fine on CPU (a few minutes per run instead of ~20 s).

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

**R_half [kpc]** (with uncertainties) is required — the model
conditions on it. Center, systemic velocity, and distance default to
data-driven medians when blank; proper motions are only needed for the
perspective-rotation correction. Systems in the bundled
local_volume_database snapshot can be prefilled by key (which also
enables the literature Wolf-mass marker).

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

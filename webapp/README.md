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

## Comparing several runs

Add up to 8 datasets; every one is overplotted on the same panels and
gets its own corner plot and summary. Each dataset past the first
chooses two things independently:

- **Catalog file** — its own upload, or another dataset's. Sharing the
  file while cutting it differently is how you isolate a *selection
  effect*.
- **Match another dataset** — when set, every field except the model
  and the run label is copied from that dataset and locked, so the
  model is the only free variable. That is the controlled *model
  comparison*.

Both relations may chain (C matches B, B matches A) and are resolved to
the root; the dropdowns omit any choice that would close a loop, so a
cycle cannot be selected. Removing a dataset resets anything pointing
at it back to independent rather than silently re-pointing it at other
data, and takes its curve and corner off screen.

Curve colors come from a **Color scheme** dropdown in the display
options — matplotlib's `tab10`, `Set1`, `Set2`, `Dark2`, `Paired` and
seaborn's `deep`, `muted`, `bright`, `dark`, `pastel`, `colorblind`,
with hexes read out of the installed matplotlib/seaborn rather than
transcribed, so a curve here matches the same series in a notebook.
The very light sets (`Set3`, `Accent`) are deliberately absent: these
panels overlay translucent credible bands and a pale line disappears
under them. The per-dataset swatch still overrides any individual
color, and now survives a re-run; picking a scheme clears the
overrides. Colors are keyed to a dataset's fixed id, so removing one
never repaints the others mid-comparison.

With more than one run on screen, **Download all posteriors** returns a
single CSV of every run stacked, with `label` and `model` columns
prepended so a `groupby` recovers whichever axis you varied. The
per-parameter columns are unchanged from the single-run download.

## Saving and restoring a configuration

**Save settings** writes a JSON file describing every input that
decides a run: each dataset's system metadata, cuts, row filter,
column assignment, flag cuts, manual star selection, model and
sampling settings, plus the dataset topology (which datasets share a
catalog or match another) and the display options. **Load settings**
puts it all back.

The catalog itself is deliberately not in the file — it can be
hundreds of MB, and the point is to record what was *done to* a
catalog. Each dataset does record the `upload_id` the server gave it,
and loading tries that first: against the same server the catalog is
picked straight back up with no re-upload, and an upload still on disk
is found even if it has aged out of the in-memory cache or the process
has since restarted against a persistent `--output-dir`. A settings
file that has travelled to another machine simply misses, which is
expected rather than an error — the dataset says so and every other
setting is applied regardless. Each dataset also records the file name
and row count it was configured against, so the app can tell you
exactly which files to re-upload. Column assignment and flag
cuts are reapplied automatically once the file arrives. If the file
does not match what was saved, the column assignment is still restored
where the names line up but the **manual star selection is dropped** —
those are row ids, and against a different file they select different
stars — and the dataset says so.

Saved dataset ids are remapped onto fresh ones in order (a file saved
with A, C, D loads as A, B, C), with every "same catalog as" and
"match" reference rewritten to suit, so the restored configuration is
the one that was saved even though the ids differ. Round-tripping is
exact: the request payload after a reload is byte-identical to the one
before it.

## Derived constraints

A second figure below the profile panels histograms five single-radius
summaries of the same posterior, with its own controls (bins, columns,
row height, fill/step, median lines) independent of the profile panels
— only the series colors are shared, so a dataset looks the same
everywhere on the page:

| panel | radius |
|---|---|
| `M(r½)` | (4/3)·R_half |
| `ρ(r½)` | (4/3)·R_half |
| `Γ(r½) = dlnρ/dlnr` | (4/3)·R_half |
| `ρ(150 pc)` | fixed 150 pc |
| `Γ(150 pc)` | fixed 150 pc |

r½ is the *deprojected* half-light radius, which is the radius Wolf et
al. 2010 write M½ at — that is why the Wolf estimate can be drawn on
the mass panel (dotted line + band) as a like-for-like comparison
rather than an approximate one. ρ₁₅₀ and Γ₁₅₀ sit at a fixed physical
150 pc, the scale the literature uses to compare dwarfs of different
sizes.

The log slope is `GeneralizedOMJeans.rho_log_slope`, analytic for the
generalized-NFW form, so it is not a finite difference off the plotted
grid. Unlike the profile bands these scalars are computed for **every**
posterior draw rather than the `n_profile_samples` subsample: they are
closed-form and cost ~0.014 ms a draw, against the Jeans integrals a
profile row needs.

**Download all figures (zip)** saves what is currently on screen — the
profile panels, the derived-constraint panels, and one corner per
dataset — as PNGs. The archive is built in the browser with a small
store-only zip writer rather than a library, since a packaged bundle
has no network access.

## Default axis ranges

The profile panels open on a fixed view rather than fitted to the run,
so two datasets — or two sessions — are directly comparable without
the axes having moved underneath. The numbers are anchored on the
dwarfs in the bundled local_volume_database snapshot, not chosen by
eye:

| panel | default | why |
|---|---|---|
| radius | 0.01–5 kpc | grid is 0.01–10; largest real `r_1/2` is 2.5 |
| density | 1e4–1e11 M☉/kpc³ | mean ρ(<r_1/2) spans 1e5.7–1e9.9 |
| enclosed mass | 1e3–1e10 M☉ | `M_1/2` spans 1e4.5–1e9.0 |
| anisotropy β | −0.5 to 1 | hard model bounds |
| σ_LOS | 0–30 km/s | covers all 45 measured dispersions (max 27.6) |
| κ_LOS | 0–10 | 3 is the Gaussian value |

The ρ and M windows carry about a decade of headroom each side of the
measured spread, because those panels draw the whole profile and not
just its value at `r_1/2`. β's bounds are not a choice: the prior has
`df_beta0 >= -0.499` and Osipkov-Merritt anisotropy rises to 1 at
large radius, so β cannot leave that interval. Every range is still
editable per axis, and **Autoscale** fits to the data as before.

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
- The last 8 uploads and 64 runs are cached in memory; older
  `job_id`s expire and need a re-run.
- A run is reused rather than recomputed when nothing feeding it has
  changed, keyed on the whole request payload. So re-running after
  switching one dataset's model leaves the others alone — which also
  holds their draws fixed, since the posterior is stochastic and a
  needless re-run would move their curves for no reason. The status
  line says which datasets were reused.
- `sample_posterior` cuts its draws to the prior box, so a run returns
  slightly fewer samples than requested (~95% of `n_samples`, in
  practice). If it returns *none*, the error names the two things that
  cause it: a zero-width R_half window, or a model whose declared
  radius convention is wrong.
- Models are loaded once at startup and held, so switching between
  them mid-session costs nothing; runs still queue behind the same
  lock.

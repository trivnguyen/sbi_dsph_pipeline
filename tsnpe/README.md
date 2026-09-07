# TSNPE pipeline

Truncated Sequential Neural Posterior Estimation (Deistler et al. 2022) on
top of `jgnn`. A run's model checkpoints and observational data are
tracked in a `state.json` manifest so later rounds can't use the wrong one.

## Config

One self-contained config file per run (no base/override split). Configs
are **git-ignored**: they hard-code scratch paths, wandb run ids and
target catalogs, so they are personal to whoever is running them. Copy
`configs/draco_override_example.py`, which is the tracked, documented
starting point, and edit it directly.

## Layout

```
configs/            per-run configs (git-ignored except the example)
  draco_override_example.py   tracked template - copy this
tsnpe/
  state.py       run-state manifest (state.json read/write; see below)
  target.py      load + hard-copy a target's observational data
  prior.py       fixed 8-param prior box, rstar conditioning, and the
                 radius-unit conversion (see "The prior is fixed")
  proposal.py    TSNPE truncated-proposal sampler (real ICRS observation
                 in, truncated proposal out)
  sims.py        Agama simulator + Cartesian HDF5 writer, same physics as
                 npe/simulate_8params_process_priorA.py
  model_io.py    rebuild an NPE model from a stored architecture config,
                 or a small fixed one for debug_model_config()
register_run.py     one-time: register target + round-0 model
simulate_round.py   round r >= 1: proposal sample + Agama simulate
train_round.py      round r >= 1: fine-tune round r-1 on round r's data
run_pipeline.sh      register once, then loop rounds

check_posteriors.py       CLI: round-0 (amortized) posterior/profile checks
                          over every mock catalog on disk
posterior_by_round.ipynb  notebook: posterior/profile checks across one
                          TSNPE run's rounds (non-amortized - each round's
                          own fine-tuned checkpoint)
```

`check_posteriors.py` (and, for other TSNPE runs, `posterior_by_round.ipynb`)
shares its plotting/profile-computation logic via `../plotting/` (sibling to
`tsnpe/`, alongside `npe/`) - it lives outside `tsnpe/` since none of it is
TSNPE-specific. The interactive explorer lives there too, since it isn't
tied to TSNPE either (it drives the amortized round-0 model directly):

```
../plotting/
  posterior_diagnostics.py  sample_posterior + Jeans profiles (density,
                            anisotropy, LOS dispersion/kurtosis) + corner/
                            profile plotting; check_posteriors.py's CLI and
                            posterior_explorer.ipynb are thin wrappers
                            around this
  catalog_registry.py       discovers selectable (catalog, target)
                            datasets - mock (with true params) and real
                            survey catalogs (without) - for
                            posterior_explorer.ipynb; see its docstring
                            for how to add a new dataset or source
  posterior_explorer.ipynb  notebook: interactive (ipywidgets) posterior/
                            profile explorer over any mock or real catalog,
                            against the amortized round-0 model
  tsnpe_run_registry.py     discovers existing TSNPE run directories under
                            TSNPE_RUNS_ROOT and which of their rounds are
                            trained (state.json-gated, like
                            posterior_by_round.ipynb)
  tsnpe_explorer.ipynb      notebook: interactive, non-amortized posterior/
                            profile explorer over any TSNPE run - pick a
                            run and any subset of its rounds to overlay,
                            with configurable percentile bands and colors
                            (posterior_by_round.ipynb's interactive,
                            multi-round-overlay counterpart)
```

Training data is raw Cartesian pos/vel (`tsnpe/sims.py`); sky-plane
projection happens as a pre-transform at train time. The real observation
(`tsnpe/target.py`) is the only thing that's ever ICRS — see
`tsnpe/proposal.py`'s docstring.

## Round semantics

- Round 0 is a pretrained wide-prior checkpoint, registered (not trained)
  by `register_run.py`.
- Round r >= 1 simulates a truncated proposal from round r-1's model, then
  fine-tunes round r-1's checkpoint on *only* round r's fresh simulations.
- The normalization dict and model architecture are fixed at round 0 and
  reused verbatim by every later round.

## state.json

Every script reads/writes `<run_dir>/state.json` instead of taking
checkpoint/x_obs paths as CLI flags. Paths are relative to `run_dir` and
point at hard copies, so a run directory is self-contained:

```json
{
  "seed": 0,
  "target": {"npz_path": "target/x_obs.npz", "sha256": "...", "key": "draco_1"},
  "base": {
    "checkpoint_path": "round_0/model.ckpt",
    "norm_dict_path": "round_0/norm_dict.json",
    "model_config_path": "round_0/model_config.json",
    "source": "wandb", "wandb_run_path": "sbi_dsph/8Params_WidePrior/1ijk8flq"
  },
  "rounds": {
    "1": {
      "data_path": "round_1/data.hdf5",
      "diagnostics": {"tau": -12.3, "acceptance_rate": 0.004,
                      "sampling_mode": "rejection"},
      "checkpoint_path": "round_1/jgnn-tsnpe/<run_id>/checkpoints/last.ckpt",
      "wandb_run_id": "abc123"
    }
  }
}
```

Every script no-ops against an already-registered/already-run step, so
`run_pipeline.sh` just always calls every step and resumes correctly after
a partial failure.

Within a round, `simulate_round.py` additionally caches the drawn proposal
(`proposal_phys.npy`, `posterior_phys.npy`, `diagnostics.json`,
`proposal_settings.json`). Drawing it costs a checkpoint load, a tau
calibration over `n_post_samples` and the sampling loop; the simulation
that follows is far longer and far likelier to be what hit the wall clock.
A retry reuses those files rather than redoing that work. The cache is
keyed on `proposal_settings.json` — the proposal config, checkpoint, seed
and round — so changing any of them draws fresh instead of silently
simulating a different distribution. Delete the round's `.npy` files to
force a redraw.

## Proposal sampling modes

`config.proposal.sampling_mode` picks how the tau-truncated region is
sampled. All four target the same distribution (the prior truncated to
`q >= tau`), so it stays a proper distribution and training needs no
importance correction:

- `'rejection'` (default) — rejection-sample the prior. Exact and
  duplicate-free; degrades as the truncated region shrinks.
- `'flow'` — rejection-sample the model, using `q >= exp(tau)` inside the
  region to bound the acceptance probability. Also exact and
  duplicate-free, and cheaper than `'rejection'` once
  `tau > -log(prior volume)` — i.e. in later rounds.
- `'sir'` — sampling-importance-resampling from the posterior. Same
  target, but resamples with replacement, so some returned rows are
  duplicates. Prefer `'flow'`.
- `'auto'` — measure both exact samplers on a short probe
  (`calibration_draws`) and run whichever accepts more. The crossover
  moves between rounds, so this is the setting that doesn't need
  revisiting as a run progresses.

Add these fields to your config explicitly if you want to override them
from the command line — `ml_collections` configs are locked, so
`--config.proposal.sampling_mode=auto` fails unless the field exists.

## The prior is fixed, not configured

The prior box (8 physical params, `stellar_log_rstar` conditioning
derived from the target's half-light radius) changes only with the
training set, so it's plain constants in `tsnpe/prior.py`, not a config
file. The bounds must match the `prior_min`/`prior_max` recorded in that
set's `config.<n>.json`.

### Radius units

The box always draws `dm_log_rdm` as `log10(r_dm / r_star)` — an offset
from the conditioning value — because that is the only space in which the
box is a box: `r_dm`'s kpc bounds slide with each row's own `r_star`
draw. What differs between training sets is the units the *model*
predicts, selected by `RADIUS_UNITS` in `tsnpe/prior.py`:

- `'kpc'` (priorA / `8p_ZhaoPlumCOM`, the current default) — the model
  predicts `log10(r_dm / kpc)`, so `to_kpc()` adds the conditioning value
  on the way out and `to_rstar_units()` subtracts it on the way back.
- `'rstar'` (priorB / `8p_ZhaoPlumCOM_v3`) — the model predicts
  `log10(r_dm / r_star)`, the same units the box uses. Nothing converts:
  `RSTAR_SCALED_PARAM_NAMES` is empty and both functions are the
  identity.

Switching to priorB is that one constant plus retraining round 0 on the
matching dataset — and `tsnpe/sims.py`'s simulator moved to the relative
convention, which has **not** been done yet.

`df_log_ra` is in `r_star` units under both conventions (`sims.py`
multiplies by `r_star` itself), so it never takes part in the conversion.

`tsnpe/proposal.py` applies the conversion and builds model-ready
features from the real observation.

## Debug mode

`config.pretrained.random_init = True` builds round 0 from a small fixed
architecture (`tsnpe.model_io.debug_model_config`) and a norm_dict that
doesn't depend on any real data (`tsnpe.prior.default_norm_dict`) — no
wandb, no target needed first. Posteriors/proposals from this are
meaningless; it only exercises the pipeline's plumbing.

## Usage

```bash
# Step by step:
python register_run.py    --config configs/my_run.py
python simulate_round.py  --config configs/my_run.py --config.round=1
python train_round.py     --config configs/my_run.py --config.round=1

# Or all at once, rounds 1..5:
./run_pipeline.sh --config configs/my_run.py --rounds 5

# Any ml_collections override is passed through, e.g. more sims per round:
./run_pipeline.sh --config configs/my_run.py --rounds 5 \
    --config.n_sims=2000
```

### On SLURM

```bash
./slurm/submit.sh configs/my_run.py --rounds=5
./slurm/submit.sh configs/my_run.py --rounds=5 --start-round=3
./slurm/submit.sh configs/my_run.py --rounds=5 --partition=debug --time=00:15:00
```

One job runs `run_pipeline.sh` end to end (register once, then loop
simulate+train per round) rather than one job per round — `simulate_round.py`'s
Agama simulation and `train_round.py`'s GPU training run sequentially
within a round, never concurrently, so a single job's resource allocation
covers both. Each submission gets its own log directory under
`$SCRATCH/slurm_logs/tsnpe/<config_name>/<timestamp>/` (config snapshot,
`manifest.txt`, `slurm-<jobid>.{out,err}`), indexed in
`$SCRATCH/slurm_logs/tsnpe/runs.tsv` — see `npe/README.md` for the same
convention there.

To resume a job that hit its time limit mid-round: resubmit the same
command (same `--rounds`, `--start-round` at or before the last completed
round). Every step in `run_pipeline.sh` is state.json-gated and no-ops if
already done, so this is always safe regardless of exactly where the
previous attempt stopped.

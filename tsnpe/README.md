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
  prior.py       the run's prior box, built from a dsph_sims spec and
                 pinned per run; rstar conditioning; box <-> model-unit
                 maps (see "The prior comes from the model spec")
  proposal.py    TSNPE truncated-proposal sampler (real ICRS observation
                 in, truncated proposal out)
  sims.py        round >= 1 simulation through the run's dsph_sims spec
                 (the same simulate/preprocess that built round 0) +
                 Cartesian HDF5 writer
  model_io.py    rebuild an NPE model from a stored architecture config,
                 or a small fixed one for debug_model_config()
register_run.py     one-time: register target + round-0 model
simulate_round.py   round r >= 1: proposal sample + Agama simulate
train_round.py      round r >= 1: fine-tune round r-1 on round r's data
posterior_by_round.py
                    round r >= 0: this run's posterior/profile plots for
                    every trained round + the cross-round comparison
                    (run_pipeline.sh calls it after each train)
run_pipeline.sh      register once, then loop rounds

check_posteriors.py       CLI: round-0 (amortized) posterior/profile checks
                          over every mock catalog on disk
PLAN_light_profile_conditioning.md
                          plan (not yet built) for running the 9p/10p/11p
                          specs: multi-axis light-profile conditioning
test_prior.py             self-check: Prior from spec == pinned == legacy
                          keywords, box maps invert, sims.py == the spec's
                          own simulate from the same agama seed
                          (../dsph_sims/tests/test_specs.py holds the
                          physics goldens)
test_posterior_by_round.py
                          self-check: posterior_by_round.py's round cache
                          invalidates on settings/checkpoint change, and
                          its per-round styling
posterior_by_round.ipynb  notebook: the interactive form of
                          posterior_by_round.py - same figures, run cell
                          by cell
```

`check_posteriors.py`, `posterior_by_round.py` and
`posterior_by_round.ipynb`
share their plotting/profile-computation logic via `../plotting/` (sibling to
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

## Per-round diagnostics

`posterior_by_round.py` draws each trained round's own posterior at the
run's fixed observation and writes, into `<run_dir>/diagnostics/`:

```
round_<r>_corner.png    that round's posterior, in the model's radius units
round_<r>_profiles.png  that round's 5 Jeans profiles vs. the binned data
rounds_corner.png       every round's posterior, overlaid
rounds_profiles.png     every round's profiles, overlaid (3x2, legend in
                        the spare cell)
rounds_convergence.png  marginal median +- 68% vs round, one panel per
                        parameter, + the round's tau/acceptance
rounds_summary.json     the same numbers, machine-readable
round_<r>.npz           cached draws: `posterior` + the five *_samples
                        arrays, plus the `settings` they were made with
```

`rounds_convergence.png` is the one that answers "did this round buy
anything?", which overlaid posteriors are bad at: two rounds whose bands
sit on top of each other look identical whether they agree to 1% or to
30%. Each panel is pinned to that parameter's own prior range, so the
bar's height against the panel's is the fraction of the prior the
posterior still occupies, and each title carries two numbers:

- `last/r0` - the final round's 68% width over round 0's. Near 1 means
  the rounds changed nothing.
- `prior` - the 68% width over the prior's own 68% range. Near 1 means
  the data never constrained that parameter, so no number of further
  rounds will move it.

Careful with the conditioning column (`stellar_log_rstar`): its "prior"
is the target's own +-n_sigma r_half window, not a prior the model
trained under, so its `prior` number answers a different question from
the other seven. The same table is printed to stdout, which is what
survives in a SLURM log.

Its last panel is the chi^2 of each round's predicted profiles against
the binned data (LOS dispersion, LOS kurtosis, and their sum), which is
the metric that catches what the widths miss: a round can leave every
width untouched and still move the posterior's *centre* onto (or off)
the data. The two 8p_ZhaoPlumCOM_v4 Draco runs both show widths flat to
within 6% while chi^2 moves a lot, and they move differently - the DESI
run improves monotonically (sigma 3.8 -> 1.6 per bin over two rounds),
the PACE run improves at round 1 and regresses at round 2 (4.8 -> 4.0
-> 4.7). Neither is visible in the widths or in the profile overlay.

The line is the posterior-median profile's chi^2 (the headline, and
what the printed table carries); the band is where individual draws
land, which sits above the line rather than around it, since the median
profile is smoother than any single draw. Read these as ranking rounds
against each other, not as p-values: the errors are asymmetric and
handled by the usual which-side heuristic, and `joint` sums two
profiles binned from the same stars, so their measurement errors are
correlated. The binned data itself is computed once per invocation and
shared by every round, so the round-to-round comparison is exact even
though `vdisp`'s own MCMC binning jitters a little between
invocations.

`run_pipeline.sh` calls it after each `train_round.py` (pass
`--no-plots` to skip), and it is a diagnostic, never a gate: a failure is
printed and the pipeline carries on, since the round's checkpoint is
already registered by then.

Rerunning is cheap, which is what makes calling it every round
reasonable: a round whose `round_<r>.npz` still matches its settings and
its checkpoint's size/mtime is read back instead of resampled, so round
r+1 costs one round's work plus a redraw of the two comparison figures.
`--overwrite` forces a recompute; deleting the `.npz` files does the
same.

It runs on **CPU** by default and that is the right choice - one
observation's posterior is a small amount of work, and the Jeans
profiles that follow are CPU-only regardless.

```bash
# Every trained round of a run, from its config or its directory:
python posterior_by_round.py --config configs/8p_ZhaoPlumCOM_v4/my_run.py
python posterior_by_round.py --run_dir models/8p_ZhaoPlumCOM_v4/my_run

# Rounds 0..2 only (what --config.round=2 does mid-pipeline):
python posterior_by_round.py --config configs/my_run.py --config.round=2

# Shade every round's 16-84 band, not just the last one:
python posterior_by_round.py --run_dir models/... --band_mode=all
```

`--band_mode` is the knob for the comparison figure's readability: with
five or six rounds overlaid, shading them all averages out to one grey
smear, so the default (`last`) shades only the final round and draws the
rest as median lines, with weight and opacity ramping by round. Round 0
is always grey and dashed - it is the shared pretrained base, not a
fine-tune on this target. `first_last` (base vs final) is the direct
answer to "what did the run buy?".

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

## The prior comes from the model spec

Which box, which radius convention and which simulator physics belong
together is described once, in the `dsph_sims` package (one `ModelSpec`
per simulator family / prior variant;
`python ../dsph_sims/scripts/simulate_batch.py --list` prints them). A run names its spec:

```python
config.model_spec = '8p_ZhaoPlumCOM_v4'   # the set round 0 was trained on
```

`register_run.py` turns `spec.tsnpe` into a `tsnpe.prior.Prior`, pins it
to `round_0/prior_config.json`, and **refuses the checkpoint** if its
training config disagrees: the dataset it names must have been simulated
by that spec (read from the dataset's own `config.0.json`, or the spec's
`datasets` list for old ones), its `labels` must be the prior's
parameters in order, and its `cond_labels` must be the single
conditioning column tsnpe supports. Every later round reads the pinned
file back; `simulate_round.py` simulates with the spec's own
`simulate`/`preprocess` — the functions that built the round-0 data —
mapped from model units by `spec.tsnpe.model_to_sim`.

| training set        | model_spec               | round-0 project      |
|---------------------|--------------------------|----------------------|
| `8p_ZhaoPlumCOM`, `_v2` | `8p_ZhaoPlumCOM` | `8p_ZhaoPlumCOM`   |
| `8p_ZhaoPlumCOM_v3` | `8p_ZhaoPlumCOM_v3` | `8p_ZhaoPlumCOM_v3`  |
| `8p_ZhaoPlumCOM_v4` | `8p_ZhaoPlumCOM_v4` | `8p_ZhaoPlumCOM_v4`  |

The 9/10/11-parameter specs exist (simulation, npe) but have no tsnpe
adapter yet — they condition on more than r_half — and `Prior.from_spec`
says so. A config with no `model_spec` and no legacy `config.prior` block
is priorA, so existing configs are untouched; a legacy
`prior_config.json` (`radius_units`/`sim_variant` keys) is read the same
way.

`tsnpe/prior.py` itself imports nothing from `dsph_sims`: the pinned dict
carries everything a `Prior` needs, so `plotting/`, the webapp's vendored
copy and `npe_inference` rebuild one without agama, and still construct
it with the old keywords (`Prior(radius_units='rstar', ...)`).

### Radius units

The box always holds `dm_log_rdm` as `log10(r_dm / r_star)` — an offset
from the conditioning value — because that is the only space in which
the training prior is a box: `r_dm`'s kpc bounds slide with each row's
own `r_star` draw. What differs between training sets is the units the
*model* predicts, recorded in the prior's `kpc_offset_names`:

- priorA — the model predicts `log10(r_dm / kpc)`, so `box_to_model()`
  adds the conditioning value on the way out and `model_to_box()`
  subtracts it on the way back (`radius_units == 'kpc'`).
- priorB, priorC — the model predicts `log10(r_dm / r_star)`, the same
  units the box uses; the tuple is empty and both maps are the identity
  (`radius_units == 'rstar'`).

`df_log_ra` is in `r_star` units under both conventions (the simulator
multiplies by `r_star` itself), so it never takes part in the conversion.

## Debug mode

`config.pretrained.random_init = True` builds round 0 from a small fixed
architecture (`tsnpe.model_io.debug_model_config`) and a norm_dict that
doesn't depend on any real data (`tsnpe.prior.default_norm_dict`) — no
wandb, no target needed first. Posteriors/proposals from this are
meaningless; it only exercises the pipeline's plumbing.

## Usage

```bash
# Step by step:
python register_run.py        --config configs/my_run.py
python simulate_round.py      --config configs/my_run.py --config.round=1
python train_round.py         --config configs/my_run.py --config.round=1
python posterior_by_round.py  --config configs/my_run.py --config.round=1

# Or all at once, rounds 1..5 (plots after each round):
./run_pipeline.sh --config configs/my_run.py --rounds 5
./run_pipeline.sh --config configs/my_run.py --rounds 5 --no-plots

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
previous attempt stopped — **provided `config.overwrite` is False**.
`register_run.py` runs on every submission and honours that flag by
deleting the whole `run_dir` first, finished rounds included. Set it to
True only for a deliberate from-scratch restart, never in a config you
will resubmit.

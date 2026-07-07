# TSNPE pipeline

Truncated Sequential Neural Posterior Estimation (Deistler et al. 2022) on
top of `jgnn`. A run's model checkpoints and observational data are
tracked in a `state.json` manifest so later rounds can't use the wrong one.

## Config

`configs/debug.py` is the one config file, self-contained (no base/
override split). Edit it directly to point at a different target or
change any setting.

## Layout

```
configs/debug.py   the config
tsnpe/
  state.py       run-state manifest (state.json read/write; see below)
  target.py      load + hard-copy a target's observational data
  prior.py       fixed 8-param prior box + rstar-conditioning transform
  proposal.py    TSNPE truncated-proposal sampler (real ICRS observation
                 in, truncated proposal out)
  sims.py        Agama simulator + Cartesian HDF5 writer, same physics as
                 npe/simulate_8params_process.py
  model_io.py    rebuild an NPE model from a stored architecture config,
                 or a small fixed one for debug_model_config()
register_run.py     one-time: register target + round-0 model
simulate_round.py   round r >= 1: proposal sample + Agama simulate
train_round.py      round r >= 1: fine-tune round r-1 on round r's data
run_pipeline.sh      register once, then loop rounds
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
      "diagnostics": {"tau": -12.3, "acceptance_rate": 0.004},
      "checkpoint_path": "round_1/jgnn-tsnpe/<run_id>/checkpoints/last.ckpt",
      "wandb_run_id": "abc123"
    }
  }
}
```

Every script no-ops against an already-registered/already-run step, so
`run_pipeline.sh` just always calls every step and resumes correctly after
a partial failure.

## The prior is fixed, not configured

The prior box (8 physical params, `stellar_log_r_star` conditioning
derived from the target's half-light radius) is unlikely to change, so
it's plain constants in `tsnpe/prior.py`, not a config file.

`tsnpe/proposal.py` applies the resulting tilde/conditioning
reparametrization and builds model-ready features from the real
observation.

## Debug mode

`config.pretrained.random_init = True` builds round 0 from a small fixed
architecture (`tsnpe.model_io.debug_model_config`) and a norm_dict that
doesn't depend on any real data (`tsnpe.prior.default_norm_dict`) — no
wandb, no target needed first. Posteriors/proposals from this are
meaningless; it only exercises the pipeline's plumbing. This is what
`configs/debug.py` currently uses.

## Usage

```bash
# Step by step:
python register_run.py    --config configs/debug.py
python simulate_round.py  --config configs/debug.py --config.round=1
python train_round.py     --config configs/debug.py --config.round=1

# Or all at once, rounds 1..5:
./run_pipeline.sh --config configs/debug.py --rounds 5

# Any ml_collections override is passed through, e.g. more sims per round:
./run_pipeline.sh --config configs/debug.py --rounds 5 \
    --config.n_sims=2000
```

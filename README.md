# sbi_dsph_pipeline

Simulation-based inference pipeline for dwarf spheroidal (dSph) dark
matter density profiles, using graph neural network posterior estimation
(`jgnn`). Two stages, kept in one repo because they're tightly coupled
through checkpoint/config formats — a change to one routinely needs a
matching change to the other:

1. **[`npe/`](npe/README.md)** — train a baseline amortized Neural
   Posterior Estimator on simulated wide-prior training data.
2. **[`tsnpe/`](tsnpe/README.md)** — Truncated Sequential NPE (Deistler et
   al. 2022): starting from `npe`'s checkpoint, iteratively simulate a
   truncated proposal around a real target observation and fine-tune,
   narrowing the posterior over several rounds.

See each subdirectory's own README for full usage.

## Dependency

Both stages depend on `jgnn` (GNN embeddings, the NPE model, shared
PyTorch Lightning training utilities) — a separate repo/package, not
vendored here. Install it editable (`pip install -e /path/to/jgnn`)
before running anything in either `npe/` or `tsnpe/`.

## Layout

```
npe/      baseline NPE trainer - simulate wide-prior training data, train
          the amortized model, submit to SLURM.
tsnpe/    truncated sequential NPE - register npe's checkpoint, then round
          by round: simulate a truncated proposal, fine-tune.
```

## Workflow

```bash
# 1. Simulate npe's training data (wide prior, no target observation
#    involved yet)
cd npe
python simulate_8params_process.py --n-sims 100000 \
    --output-dir /scratch/$USER/datasets/8p_ZhaoPlumCOM

# 2. Train the baseline NPE - locally, or via slurm/submit.sh on a cluster
python train_npe.py --config configs/chebconv_8params.py

# 3. Feed that checkpoint into tsnpe's truncated rounds against a real
#    target observation (config.pretrained points at npe's wandb run or
#    a local checkpoint - see tsnpe/README.md)
cd ../tsnpe
python register_run.py --config configs/draco.py
./run_pipeline.sh --config configs/draco.py --rounds 5
```

`npe` and `tsnpe` are otherwise independent to run — `tsnpe` only ever
*reads* a finished `npe` checkpoint (plus its recorded `norm_dict` and
pre-transforms config), it never modifies `npe`'s output.

# NPE pipeline

Amortized Neural Posterior Estimation on top of `jgnn`: simulate wide-prior
training data, train a GNN-embedding + normalizing-flow model, checkpoint
it for `tsnpe/` to fine-tune against a real target.

## Layout

```
simulate_8params_process_priorA.py
                              simulate training data (ProcessPoolExecutor;
                               each worker has its own isolated agama
                               state, so no locking needed - prefer this
                               one). priorA: r_dm in kpc, r_star in units
                               of r_dm, r_a in units of r_star
simulate_8params_process_priorB.py
                              same, with r_dm and r_a both in units of
                               r_star and r_star in kpc - the
                               parameterization tsnpe's RADIUS_UNITS
                               ='rstar' expects
simulate_8params_thread.py    same simulation, ThreadPoolExecutor version
                               (needs a lock around agama's RNG-touching
                               .sample() call - kept for comparison)
train_npe.py                  train the model
configs/                      per-run configs (git-ignored; copy an
                               existing one as a starting point)
slurm/
  submit.sh                   submit train_npe.py to SLURM with per-run
                               log bookkeeping
  train_npe.sbatch            the actual job script (usually launched via
                               submit.sh, not directly)
```

## Simulate training data

```bash
python simulate_8params_process_priorA.py \
    --n-sims 100000 --n-workers 24 --output-dir /scratch/$USER/datasets/8p_ZhaoPlumCOM
```

Draws from a fixed wide prior (`PRIOR_MIN`/`PRIOR_MAX` at the top of
the script) and simulates each galaxy's 6D stellar
kinematics with Agama, writing Cartesian pos/vel/vel_error to sharded
HDF5 files (`--galaxies-per-file`). `--append` resumes into an existing
output directory instead of overwriting it.

## Train

```bash
python train_npe.py --config configs/my_run.py
```

Config fields worth knowing:

- `config.workdir` — shared root across every run of every project (e.g.
  `/scratch/$USER/trained_models/npe`), **not** project-specific.
  `config.wandb_project` is the per-project name; `WandbLogger` nests
  `workdir/<wandb_project>/<run_id>/checkpoints/` on its own, so folding
  the project name into `workdir` too causes double nesting.
- `config.checkpoint = 'last.ckpt'` + `config.id = '<fixed run id>'` —
  resume this exact run. Both must be set together: `config.id` fixed is
  what makes `project_dir` (and therefore `last.ckpt`'s location)
  deterministic across resubmissions. Leave both unset to start fresh.
- `config.reset_optimizer` — `False` (default) does a full resume
  (optimizer/scheduler/epoch/RNG state all continue); `True` loads weights
  only and starts training fresh from them (use for transfer learning, not
  routine resumes).

On resume, the checkpoint's own recorded `norm_dict` is always reused
(never recomputed from data) — see `train_npe.py`'s `main()`.

### On SLURM

```bash
./slurm/submit.sh configs/my_run.py
./slurm/submit.sh configs/my_run.py --time=1-00:00:00 --partition=compute_h200
./slurm/submit.sh configs/my_run.py --config.train_batch_size=128
```

Each submission gets its own log directory under
`$SCRATCH/slurm_logs/npe/<config_name>/<timestamp>/` (config snapshot,
`manifest.txt`, `slurm-<jobid>.{out,err}`), indexed in
`$SCRATCH/slurm_logs/npe/runs.tsv`. `$HOME` is read-only on compute nodes
here, so all run artifacts and caches (`XDG_CACHE_HOME`, `TORCH_HOME`,
etc.) are redirected under `$SCRATCH` — see `train_npe.sbatch`.

To resume a job that hit its time limit: resubmit the same command. As
long as `config.id` is fixed in the config file, it picks up
`config.checkpoint` from the same run directory automatically.

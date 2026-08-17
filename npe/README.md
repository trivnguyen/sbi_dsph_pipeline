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
simulate_10params_process_flathalo.py
                              priorA + DM halo axis ratio q and viewing
                               inclination i (Multipole, lmax=24)
simulate_9params_process_plumgamma.py
                              priorA + free tracer inner slope
                               gamma_star, with beta0 <= 0
simulate_11params_process_abg.py
                              priorA with the Plummer tracer replaced by
                               a full alpha-beta-gamma profile
train_npe.py                  train the model
eval_utils.py                 load a trained run, sample its posterior
                               over the held-out shard, and make the
                               corner / pred-v-true / calibration plots
eval_10params_flathalo.ipynb
eval_9params_plumgamma.ipynb
eval_11params_abg.ipynb       held-out test of each extension run
study_flattened_halo_qi.ipynb
                              simulation study: extend priorA with a DM
                               halo axis ratio q and a viewing
                               inclination i (10 parameters). Consistency
                               checks + a drop-in simulator
study_abg_light_profile.ipynb
                              simulation study: replace the Plummer
                               tracer with an alpha-beta-gamma light
                               profile (11 parameters). Consistency
                               checks + a drop-in simulator
configs/                      per-run configs (git-ignored; copy an
                               existing one as a starting point)
slurm/
  submit.sh                   submit train_npe.py to SLURM with per-run
                               log bookkeeping
  train_npe.sbatch            the actual job script (usually launched via
                               submit.sh, not directly)
  simulate.sbatch             run a simulator on the Trillium CPU
                               cluster - submit from tri-login01, not
                               from the GPU login node
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

## Prior extension studies

`study_flattened_halo_qi.ipynb` and `study_abg_light_profile.ipynb` are two
**independent** proposals for widening the 8-parameter prior, each built on
`simulate_8params_process_priorA.py`. Both are simulation studies only —
they validate that the extended simulator produces what its labels claim,
and do not attempt inference. Each ends with a self-contained drop-in
replacement for `simulator()` plus the prior box it needs.

Two findings apply to the **current** pipeline regardless of whether either
extension is adopted:

- Agama's `QuasiSpherical` DF silently clips a negative Eddington inversion
  to zero — no exception, no warning — whenever the slope–anisotropy
  theorem `gamma_star >= 2 * beta0` is violated. Plummer has
  `gamma_star = 0`, so every `beta0 > 0` in `PRIOR_A` is in that regime;
  measured, essentially every galaxy above `beta0 = 0.5` has a stellar
  density that departs from its own Plummer label by more than 0.1 dex,
  with the median error climbing to of order a decade at the top of the
  range. The `(theta, x)` pairs stay self-consistent, so existing
  posteriors are not invalid — but `r_star` and `beta0` stop meaning what
  their names say over roughly half the prior.
- The same theorem has a *local* form that bites when `r_a << r_star`,
  which `log_ra >= -1` currently allows.

See section 8c of `study_abg_light_profile.ipynb` for the measurement.

### The three extension runs

Each study was turned into a simulator plus a training config. All three use
priorA units, 200k galaxies at Poisson(100) stars, and keep the same seven
DM/DF labels as the 8-parameter baseline so the posteriors are comparable.

| simulator | dataset | config | new parameters |
|---|---|---|---|
| `simulate_10params_process_flathalo.py` | `10p_flathalo_qi` | `configs/chebconv_10params_flathalo.py` | `dm_q`, `cos_inc` (labels) |
| `simulate_9params_process_plumgamma.py` | `9p_plumgamma` | `configs/chebconv_9params_plumgamma.py` | `stellar_gamma` (conditioning) |
| `simulate_11params_process_abg.py` | `11p_abg` | `configs/chebconv_11params_abg.py` | `stellar_{alpha,beta,gamma}` (conditioning) |

`q` and `cos_inc` are inference targets because neither is observable. The
light-profile shape parameters are conditioning inputs instead, because
photometry measures them — the same reasoning that already makes the
half-light radius a conditioning input. Moving them into `labels` turns each
run into a test of whether the kinematics alone constrain the light profile;
it is a one-line change in the config.

The two tracer-shape runs condition on `stellar_log_rhalf_kpc` rather than
`stellar_log_rstar_kpc`. For Plummer the two coincide, but once the shape is
free the scale radius stops being an observable: `R_h/r_star` spans about
0.8-6 in `9p_plumgamma` and more than two decades in `11p_abg`. The
simulators compute `R_h` per galaxy from the analytic projected profile.

Simulation runs on the CPU cluster, training on the GPU cluster — they are
separate schedulers, so a SLURM dependency cannot chain them:

```bash
ssh tri-login01
cd .../npe/slurm
sbatch --job-name=sim_11p_abg simulate.sbatch \
    simulate_11params_process_abg.py /scratch/$USER/datasets/11p_abg 240000 \
    --n-stars 100 --galaxies-per-file 25000 --seed 810233
```

`n_sims` is set above 200k to absorb the 85-90% acceptance rate;
`--galaxies-per-file 25000` with `config.num_datasets = 8` then selects
exactly 200k galaxies and ignores the partial final shard.

### Evaluating a run

Each simulator was run long enough to leave a spare shard, and
`config.num_datasets = 8` reads shards 0-7, so `data.8.h5` is a genuine
held-out test set rather than a re-split of training data. `eval_utils.py`
defaults to that shard.

```python
import eval_utils as ev
model, config, norm_dict = ev.load_npe(ev.find_run_dir('11p_abg'))
loader, shard = ev.load_test_set(config, norm_dict, max_graphs=2000)
samples, truth = ev.sample_posterior(model, loader, num_samples=1000)
```

Two things `eval_utils` handles that are easy to get wrong by hand:

- `NPE.load_from_checkpoint` alone is **not** enough. `embedding_nn` and
  `pre_transforms` are excluded from the saved hyperparameters, so a bare
  load silently gives an identity embedding and plausible-looking garbage.
  `load_npe` rebuilds both from `config_snapshot.json`.
- Posterior samples and `theta` live in the normalized space the flow was
  trained in. `sample_posterior` un-normalizes both back to physical units
  using the checkpoint's own `norm_dict`, never a recomputed one.

The diagnostics answer different questions and are worth reading together:

- **TARP** — coverage of the *joint* posterior, condensed to one curve.
- **Rank histograms** — each *marginal* separately.
- **Correlation checks** — the pairwise structure, which the two above can
  both miss. `plot_correlation_matrices` whitens each galaxy's error by its
  own posterior covariance (identity means the reported correlation is
  right); `plot_pairwise_rank_calibration` is the distribution-free
  version, testing rank uniformity along `theta_i - theta_j` and
  `theta_i + theta_j` rather than along the coordinate axes alone.
- **MIRA** — calibration as a single number.

The correlation checks are not redundant. On a conjugate-Gaussian control
whose true posterior correlation was deliberately zeroed, the 1-D rank
histograms stayed at reduced chi-square 0.70 and 1.45 — perfectly healthy —
while the difference-direction test rose to 10.0, and to 26.1 when the
correlation was flipped in sign.

None of these measures informativeness: a posterior that just returns the
prior is perfectly calibrated and scores well on all of them. The mean 68%
credible width divided by the prior range, printed in section 3 of each
notebook, is what says whether the posterior is actually useful.

One operational note: run anything numpy-heavy here with
`OMP_NUM_THREADS` set to a small number. On a login node the default fans
out to all 192 cores and burns the 3600 CPU-second budget at 192x the wall
rate, which kills the process well before it looks slow.

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

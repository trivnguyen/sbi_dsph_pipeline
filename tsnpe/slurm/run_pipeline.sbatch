#!/bin/bash
# run_pipeline.sbatch — SLURM job script for run_pipeline.sh on Trillium (SciNet).
#
# Normally launched via ./submit.sh <config.py>; can also run directly:
#   sbatch run_pipeline.sbatch configs/draco.py --rounds 5
#
# $HOME is read-only on compute nodes here, so all writes (logs, manifest,
# caches) go under /scratch/tvnguyen. No --mem requested: it's a no-op on
# this cluster's GPU partitions (DefMemPerNode == MaxMemPerNode). Single
# GPU only — train_round.py's Trainer has no devices/strategy set, so more
# than one GPU would trigger unintended DDP. simulate_round.py's Agama
# simulation (config.simulation.n_jobs) and train_round.py's GPU work run
# sequentially within a round (see run_pipeline.sh), not concurrently, so
# one job's resource allocation covers both.
#
# run_pipeline.sh's own steps are all idempotent (state.json-gated), so a
# job that hits the time limit mid-round can just be resubmitted with the
# same --rounds - already-finished rounds/steps no-op and it picks up
# where it left off.

#SBATCH --job-name=tsnpe_pipeline
#SBATCH --account=def-tingli
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tri.vt.nguyen@gmail.com
#SBATCH --chdir=/scratch/tvnguyen
#SBATCH --output=/scratch/tvnguyen/slurm_logs/tsnpe/manual/%x_%j.out
#SBATCH --error=/scratch/tvnguyen/slurm_logs/tsnpe/manual/%x_%j.err

set -uo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch run_pipeline.sbatch <path/to/config.py> [--rounds N] [--start-round R] [--config.field=value ...]" >&2
    exit 1
fi

CONFIG="$1"
shift

ROUNDS=1
START_ROUND=1
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rounds)      ROUNDS="$2";      shift 2 ;;
        --start-round) START_ROUND="$2"; shift 2 ;;
        *)             EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Not self-located via $BASH_SOURCE: SLURM runs a copy of this script from
# a spool dir, so that would resolve to the wrong directory. submit.sh
# exports TSNPE_DIR correctly; this fallback is only for direct `sbatch` use.
TSNPE_DIR="${TSNPE_DIR:-/home/tvnguyen/projects/sbi_dsph/sbi_dsph_pipeline/tsnpe}"
RUN_DIR="${RUN_DIR:-/scratch/tvnguyen/slurm_logs/tsnpe/manual/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
MANIFEST="$RUN_DIR/manifest.txt"
mkdir -p "$RUN_DIR"

# Resolve config relative to tsnpe/ if it isn't already absolute/relative-found.
if [[ ! -f "$CONFIG" ]]; then
    if [[ -f "$TSNPE_DIR/$CONFIG" ]]; then
        CONFIG="$TSNPE_DIR/$CONFIG"
    fi
fi

# One OpenMP thread per process. simulate_round.py runs a pool of
# cpus-per-task worker processes, so anything more multiplies out to
# hundreds of threads on 24 CPUs and runs slower than serial. Set here as
# well as in simulate_round.py so it also covers whatever agama/BLAS the
# other two steps touch. train_round.py sizes torch's own pool from the
# job's CPU affinity, which this does not constrain.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Redirect caches that default to $HOME (read-only on compute nodes).
export XDG_CACHE_HOME="/scratch/tvnguyen/cache"
export MPLCONFIGDIR="/scratch/tvnguyen/cache/matplotlib"
export WANDB_CONFIG_DIR="/scratch/tvnguyen/cache/wandb_config"
export TORCH_HOME="/scratch/tvnguyen/cache/torch"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$WANDB_CONFIG_DIR" "$TORCH_HOME"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MANIFEST"; }

log "===== job start ====="
log "job_id:       $SLURM_JOB_ID"
log "job_name:     $SLURM_JOB_NAME"
log "node:         $(hostname)"
log "partition:    ${SLURM_JOB_PARTITION:-unknown}"
log "gpus:         $(nvidia-smi -L 2>/dev/null | tr '\n' ';' )"
log "cpus:         ${SLURM_CPUS_PER_TASK:-unknown}"
log "mem:          ${SLURM_MEM_PER_NODE:-unknown} MB"
log "config:       $CONFIG"
log "rounds:       $START_ROUND .. $ROUNDS"
log "extra_args:   ${EXTRA_ARGS[*]:-<none>}"
log "jgnn commit:  $(git -C /home/tvnguyen/projects/sbi_dsph/jgnn rev-parse --short HEAD 2>/dev/null || echo unknown)"

cp -- "$CONFIG" "$RUN_DIR/config_snapshot.py" 2>/dev/null || true

# Activate the venv directly rather than via ~/.bashrc's venv-activate:
# ~/.bashrc's non-interactive guard skips its own function definitions
# under a non-interactive sbatch shell, so venv-activate wouldn't exist.
VENV_DIR="$HOME/.venvs/torch"
set +u  # Lmod's `module load` internals aren't -u-safe.
if [[ -s "$VENV_DIR/.modules" ]]; then
    module load $(cat "$VENV_DIR/.modules") 2>/dev/null
fi
source "$VENV_DIR/bin/activate"
set -u
log "python:       $(which python3) ($(python3 --version 2>&1))"
log "torch:        $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"

on_exit() {
    local code=$?
    log "exit_code:    $code"
    log "===== job end ====="
    exit "$code"
}
trap on_exit EXIT
trap 'log "received TERM (likely time limit approaching)"' TERM

cd "$TSNPE_DIR"
./run_pipeline.sh --config "$CONFIG" --rounds "$ROUNDS" --start-round "$START_ROUND" \
    "${EXTRA_ARGS[@]}" --config.training.enable_progress_bar=False 2>&1 | tee -a "$RUN_DIR/stdout.log"
exit "${PIPESTATUS[0]}"

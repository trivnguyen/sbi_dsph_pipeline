#!/bin/bash
# simulate.sbatch — SLURM job script for dsph_sims/scripts/simulate_batch.py
#   (pass --model <spec> as an extra arg) or any simulate_*.py
# simulators, on the Trillium CPU cluster (SciNet).
#
# These are pure-CPU, embarrassingly parallel jobs, so they belong on the
# CPU cluster (tri-login01, ClusterName=trillium), not on the GPU cluster
# that train_npe.sbatch targets. Submit from a CPU login node:
#
#   ssh tri-login01
#   sbatch simulate.sbatch <simulator.py> <output_dir> <n_sims> [extra args]
#
# The login nodes enforce a 3600 CPU-second limit, which even the cheapest
# of these runs blows through in seconds at full width — always sbatch.
#
# $HOME is read-only on compute nodes, so everything is written under
# /scratch/tvnguyen.

#SBATCH --job-name=npe_sim
#SBATCH --account=def-tingli
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=03:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tri.vt.nguyen@gmail.com
#SBATCH --chdir=/scratch/tvnguyen
#SBATCH --output=/scratch/tvnguyen/slurm_logs/npe/sim/%x_%j.out
#SBATCH --error=/scratch/tvnguyen/slurm_logs/npe/sim/%x_%j.err

set -uo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: sbatch simulate.sbatch <simulator.py> <output_dir> <n_sims> [extra args]" >&2
    exit 1
fi

SIM_SCRIPT="$1"
OUTPUT_DIR="$2"
N_SIMS="$3"
shift 3
EXTRA_ARGS=("$@")

NPE_DIR="${NPE_DIR:-/home/tvnguyen/projects/sbi_dsph/sbi_dsph_pipeline/npe}"
if [[ ! -f "$SIM_SCRIPT" ]]; then
    SIM_SCRIPT="$NPE_DIR/$SIM_SCRIPT"
fi

# Redirect caches that default to $HOME (read-only on compute nodes).
export XDG_CACHE_HOME="/scratch/tvnguyen/cache"
export MPLCONFIGDIR="/scratch/tvnguyen/cache/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

# One OpenMP thread per worker process: the simulators fan out over
# processes, so any threading inside agama would only oversubscribe.
export OMP_NUM_THREADS=1

echo "===== job start ====="
echo "job_id:     ${SLURM_JOB_ID:-none}"
echo "node:       $(hostname)"
echo "cluster:    $(scontrol show config | awk '/ClusterName/{print $3}')"
echo "cpus:       ${SLURM_CPUS_PER_TASK:-unknown}"
echo "simulator:  $SIM_SCRIPT"
echo "output_dir: $OUTPUT_DIR"
echo "n_sims:     $N_SIMS"
echo "extra:      ${EXTRA_ARGS[*]:-<none>}"

VENV_DIR="$HOME/.venvs/torch"
set +u  # Lmod's `module load` internals aren't -u-safe.
if [[ -s "$VENV_DIR/.modules" ]]; then
    module load $(cat "$VENV_DIR/.modules") 2>/dev/null
fi
source "$VENV_DIR/bin/activate"
set -u
echo "python:     $(which python3) ($(python3 --version 2>&1))"

cd "$NPE_DIR"
python3 "$SIM_SCRIPT" \
    --n-sims "$N_SIMS" \
    --n-workers "${SLURM_CPUS_PER_TASK:-192}" \
    --output-dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"
CODE=$?

echo "exit_code:  $CODE"
echo "===== job end ====="
exit "$CODE"

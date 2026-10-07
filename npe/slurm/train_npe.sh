#!/bin/bash
# train_npe.sh — local (no-SLURM) equivalent of slurm/train_npe.sbatch for tara.
#
#   ./slurm/train_npe.sh configs/chebconv_8params_v4.py              # detached (default)
#   ./slurm/train_npe.sh -f configs/chebconv_8params_v4.py           # stay in this shell
#   ./slurm/train_npe.sh --time 12h configs/chebconv_8params_v4.py   # wall-clock limit
#   ./slurm/train_npe.sh configs/chebconv_8params_v4.py --config.batch_size=256
#
# Mirrors the sbatch script's behaviour: a per-run directory with a manifest,
# a config snapshot, and separate stdout/stderr logs. Unlike SLURM there is no
# queue -- the run starts immediately and competes for the single GB10 GPU.

set -uo pipefail

JOB_NAME="${JOB_NAME:-npe_train}"
NPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${LOG_ROOT:-$HOME/scratch/local_logs/npe}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/torch}"
BACKGROUND=1          # detached by default, like sbatch
TIME_LIMIT=""

usage() {
    cat >&2 <<'USAGE'
Usage: train_npe.sh [options] <path/to/config.py> [--config.field=value ...]

  -f, --foreground      run in this shell instead of detaching
  -b, --background      run detached (default)
  -t, --time DURATION   wall-clock limit, e.g. 12h, 90m, 3600s
  -h, --help            show this message

Detached is the default: the run survives logout, as an sbatch job would.
USAGE
    exit 1
}

# Hand-rolled parser: getopts cannot do long options, and we must not
# consume the --config.* overrides meant for train_npe.py.
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--foreground) BACKGROUND=0; shift ;;
        -b|--background) BACKGROUND=1; shift ;;
        -t|--time)
            [[ $# -ge 2 ]] || { echo "$1 needs a DURATION" >&2; usage; }
            TIME_LIMIT="$2"; shift 2 ;;
        --time=*)        TIME_LIMIT="${1#*=}"; shift ;;
        -t=*)            TIME_LIMIT="${1#*=}"; shift ;;
        -h|--help)       usage ;;
        --)              shift; break ;;
        *)               break ;;   # config path; the rest passes through
    esac
done
[[ $# -lt 1 ]] && usage

CONFIG="$1"; shift
EXTRA_ARGS=("$@")

# Resolve the config relative to npe_dsph/ if not found as given.
if [[ ! -f "$CONFIG" ]]; then
    if [[ -f "$NPE_DIR/$CONFIG" ]]; then
        CONFIG="$NPE_DIR/$CONFIG"
    else
        echo "Config not found: $CONFIG" >&2
        exit 1
    fi
fi
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

# Stand-in for $SLURM_JOB_ID: timestamp + PID is unique and sorts chronologically.
RUN_ID="$(date +%Y%m%d-%H%M%S)_$$"
RUN_DIR="$LOG_ROOT/${JOB_NAME}_${RUN_ID}"
MANIFEST="$RUN_DIR/manifest.txt"
mkdir -p "$RUN_DIR" || { echo "Cannot create $RUN_DIR" >&2; exit 1; }

# Re-exec detached once the run dir exists, so the caller gets the real paths.
if [[ "$BACKGROUND" == 1 ]]; then
    export JOB_NAME LOG_ROOT VENV_DIR
    RUN_DIR_PRESET="$RUN_DIR" \
        nohup "$0" --foreground ${TIME_LIMIT:+--time "$TIME_LIMIT"} "$CONFIG" "${EXTRA_ARGS[@]}" \
        > "$RUN_DIR/nohup.out" 2>&1 &
    echo "Started detached: PID $!"
    echo "  run dir: $RUN_DIR"
    echo "  follow:  tail -f $RUN_DIR/stdout.log"
    exit 0
fi
RUN_DIR="${RUN_DIR_PRESET:-$RUN_DIR}"
MANIFEST="$RUN_DIR/manifest.txt"
mkdir -p "$RUN_DIR"

# Caches that would otherwise land in $HOME. Kept on scratch to match trillium,
# where $HOME was read-only on compute nodes.
export XDG_CACHE_HOME="$HOME/scratch/cache"
export MPLCONFIGDIR="$HOME/scratch/cache/matplotlib"
export WANDB_CONFIG_DIR="$HOME/scratch/cache/wandb_config"
export TORCH_HOME="$HOME/scratch/cache/torch"
export NUMBA_CACHE_DIR="$HOME/scratch/cache/numba"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$HOME/scratch/wandb_data}"
export WANDB_X_SERVICE_WAIT=300
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$WANDB_CONFIG_DIR" "$TORCH_HOME" "$NUMBA_CACHE_DIR" "$WANDB_DATA_DIR"

# One GPU, and threads capped so a run does not take all 20 cores.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONPATH="${PYTHONPATH:-}:$HOME/my_modules"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MANIFEST"; }

log "===== run start ====="
log "run_id:       $RUN_ID"
log "job_name:     $JOB_NAME"
log "host:         $(hostname)"
log "pid:          $$"
log "gpus:         $(nvidia-smi -L 2>/dev/null | tr '\n' ';')"
log "cpus:         $(nproc --all) total (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
log "mem:          $(free -g | awk '/^Mem:/{print $2" GB total, "$7" GB available"}')"
log "config:       $CONFIG"
log "extra_args:   ${EXTRA_ARGS[*]:-<none>}"
log "time_limit:   ${TIME_LIMIT:-<none>}"
log "wandb_data:   $WANDB_DATA_DIR (mode=${WANDB_MODE:-online})"

cp -- "$CONFIG" "$RUN_DIR/config_snapshot.py" 2>/dev/null || true

# Activate directly: ~/.bashrc's venv-activate is not defined in a
# non-interactive shell, same reason as on trillium.
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    log "FATAL: no venv at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"
log "python:       $(command -v python3) ($(python3 --version 2>&1))"
log "torch:        $(python3 -c 'import torch; print(torch.__version__, "cuda:", torch.cuda.is_available())' 2>/dev/null)"

on_exit() {
    local code=$?
    log "exit_code:    $code"
    log "===== run end ====="
    exit "$code"
}
trap on_exit EXIT
trap 'log "received TERM -- shutting down"' TERM
trap 'log "received INT (Ctrl-C)"' INT

cd "$NPE_DIR" || exit 1

RUNNER=(python3 train_npe.py --config "$CONFIG" "${EXTRA_ARGS[@]}" --config.enable_progress_bar=False)
if [[ -n "$TIME_LIMIT" ]]; then
    RUNNER=(timeout --signal=TERM --kill-after=60 "$TIME_LIMIT" "${RUNNER[@]}")
fi

log "command:      ${RUNNER[*]}"
"${RUNNER[@]}" > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2)
exit "${PIPESTATUS[0]}"

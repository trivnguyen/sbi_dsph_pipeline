#!/usr/bin/env bash
# run_pipeline.sh — Run the full TSNPE pipeline for a range of rounds.
#
# register_run.py and each round's scripts are all idempotent: they check
# state.json and no-op (or, for a round already checkpointed, skip) if
# that step already ran. So this script never needs its own "has this
# already happened?" logic — it just always calls every step.
#
# Usage
# -----
#   ./run_pipeline.sh --config configs/debug.py --rounds 5
#
#   # Resume a partially-completed run (already-done steps are skipped):
#   ./run_pipeline.sh --config configs/debug.py --rounds 5
#
#   # Extra ml_collections overrides are passed through to every script:
#   ./run_pipeline.sh --config configs/debug.py --rounds 5 \
#       --config.n_sims=2000
#
# Options
# -------
#   --config       PATH   ml_collections config file                 [required]
#   --rounds       N      Last round to run, inclusive                [default: 1]
#   --start-round  R      First round to run                          [default: 1]
#   (any other --config.<field>=<value> flags are forwarded as-is)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG=""
ROUNDS=1
START_ROUND=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2";       shift 2 ;;
        --rounds)       ROUNDS="$2";       shift 2 ;;
        --start-round)  START_ROUND="$2";  shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required."
    exit 1
fi

echo "=============================="
echo "  TSNPE pipeline"
echo "  config:      $CONFIG"
echo "  rounds:      $START_ROUND .. $ROUNDS"
echo "=============================="

echo ""
echo "[register] target + round-0 model"
python3 "$SCRIPT_DIR/register_run.py" --config "$CONFIG" "${EXTRA_ARGS[@]}"

for r in $(seq "$START_ROUND" "$ROUNDS"); do
    echo ""
    echo "=============================="
    echo "  Round $r / $ROUNDS"
    echo "=============================="

    echo "[simulate] round $r"
    python3 "$SCRIPT_DIR/simulate_round.py" \
        --config "$CONFIG" --config.round="$r" "${EXTRA_ARGS[@]}"

    echo "[train] round $r"
    python3 "$SCRIPT_DIR/train_round.py" \
        --config "$CONFIG" --config.round="$r" "${EXTRA_ARGS[@]}"

    echo "Round $r complete."
done

echo ""
echo "All rounds complete."

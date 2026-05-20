#!/usr/bin/env bash
# Convenience wrapper: run run_v6.sh with seeds 0/1/2 sequentially,
# then aggregate into mean ± std tables.
#
# To use different seed list:  SEEDS="0 1 2 3 4" bash src/run_all_seeds.sh

set -e
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

SEEDS="${SEEDS:-0 1 2}"

for s in $SEEDS; do
    echo ""
    echo "######################################################################"
    echo "#                       SEED  =  $s"
    echo "######################################################################"
    SEED=$s bash src/run_v6.sh
done

# Build the --seed_dirs argument
SEED_DIR_ARGS=""
for s in $SEEDS; do
    SEED_DIR_ARGS="$SEED_DIR_ARGS outputs/seed_$s"
done

echo ""
echo "######################################################################"
echo "#  AGGREGATING ACROSS SEEDS:$SEED_DIR_ARGS"
echo "######################################################################"

python -m src.eval.compare_runs_with_seeds \
    --seed_dirs $SEED_DIR_ARGS \
    --out_csv outputs/main_table_meanstd.csv \
    --ablation_csv outputs/ablation_table_meanstd.csv

echo ""
echo "===================================================================="
echo " ALL SEEDS DONE — final tables:"
echo "  outputs/main_table_meanstd.csv"
echo "  outputs/ablation_table_meanstd.csv"
echo "===================================================================="

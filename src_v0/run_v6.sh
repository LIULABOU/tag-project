#!/usr/bin/env bash
# =============================================================================
#  run_v6.sh — end-to-end pipeline (single training seed)
#
#  ENV VARS:
#    SEED      training seed (model init, dataloader shuffle).  default=42
#              The data split is INDEPENDENT of $SEED -- it is created once
#              with SPLIT_SEED=42 and reused across all training seeds.
#
#  Multi-seed usage:
#    for s in 0 1 2; do SEED=$s bash src/run_v6.sh; done
#  then aggregate:
#    python -m src.eval.compare_runs_with_seeds \
#        --seed_dirs outputs/seed_0 outputs/seed_1 outputs/seed_2 \
#        --out_csv outputs/main_table_meanstd.csv \
#        --ablation_csv outputs/ablation_table_meanstd.csv
#  Or just:
#    bash src/run_all_seeds.sh
# =============================================================================

set -e
set -u
set -o pipefail

# Auto-cd to project root (parent of src/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ----------------------------- Config ----------------------------------------
SEED="${SEED:-42}"
SPLIT_SEED=42                     # hardcoded; split is shared across SEED runs

SPLIT_MANIFEST="outputs/split_v1.json"     # written here -- data/ may be read-only
SHARDS_DIR="data/photochat/train_json"
IMAGE_MAP="data/photochat/train_image_photo_desc.jsonl"

OUT_BASE="outputs/seed_${SEED}"

EPOCHS=12
BATCH_SIZE=16
LR=5e-4
N_TEST_SHARDS=2

# Common loss settings
ALPHA_STAB=0.1
BETA_SWITCH=0.1
TARGET_RATE=0.15
GAMMA_RATE=10.0
DCP_INIT_A=2.0
TAU_FLOOR=0.20
GAMMA_FLOOR=0.1
EPSILON_DROP=0.05
GAMMA_DROP=0.1

mkdir -p "$OUT_BASE"

echo ""
echo "######################################################################"
echo "#  TRAINING SEED = $SEED   (split fixed @ seed=$SPLIT_SEED)"
echo "#  outputs:        $OUT_BASE"
echo "######################################################################"

# ----------------------------- Step 0: split ---------------------------------
echo ""
echo "===== STEP 0: split manifest ====="
if [ -f "$SPLIT_MANIFEST" ]; then
    echo "[skip] $SPLIT_MANIFEST already exists"
else
    python -m src.dataloaders.make_split \
        --shards_dir "$SHARDS_DIR" \
        --n_test "$N_TEST_SHARDS" \
        --seed "$SPLIT_SEED" \
        --out "$SPLIT_MANIFEST"
fi

# ----------------------------- helper: train ---------------------------------
run_train() {
    local NAME="$1"
    local GPU="$2"
    shift 2
    local OUT_DIR="$OUT_BASE/run_${NAME}_v6"
    if [ -f "$OUT_DIR/model.pt" ]; then
        echo "[skip train] $NAME (model.pt exists in $OUT_DIR)"
        return
    fi
    echo "[train] $NAME on GPU $GPU -> $OUT_DIR"
    CUDA_VISIBLE_DEVICES="$GPU" python -m src.train.train_mvp_wa_wh \
        --split_manifest "$SPLIT_MANIFEST" \
        --image_map_jsonl "$IMAGE_MAP" \
        --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LR" --seed "$SEED" \
        --out_dir "$OUT_DIR" \
        "$@"
}

# ----------------------------- Step 1: training ------------------------------
echo ""
echo "===== STEP 1: train 6 variants (2 waves x 3 GPU) ====="

# ---- Wave 1: fixed / dcp_B / pure_corr ----
run_train fixed 0 \
    --gate_variant fixed --rho 0.1 \
    --alpha_stab "$ALPHA_STAB" --beta_switch "$BETA_SWITCH" &

run_train dcp 1 \
    --gate_variant dcp_B \
    --target_repair_rate "$TARGET_RATE" --gamma_rate "$GAMMA_RATE" \
    --dcp_init_a "$DCP_INIT_A" \
    --alpha_stab "$ALPHA_STAB" --beta_switch "$BETA_SWITCH" &

run_train pure_corr 2 \
    --gate_variant fixed --rho 0.1 \
    --alpha_stab 0.0 --beta_switch 0.0 &

wait
echo "[wave 1 done]"

# ---- Wave 2: multi_C / multi_C+floor / multi_C+floor+drop ----
run_train multi_C 0 \
    --gate_variant multi_C \
    --target_repair_rate "$TARGET_RATE" --gamma_rate "$GAMMA_RATE" \
    --alpha_stab "$ALPHA_STAB" --beta_switch "$BETA_SWITCH" &

run_train multi_C_floor 1 \
    --gate_variant multi_C \
    --target_repair_rate "$TARGET_RATE" --gamma_rate "$GAMMA_RATE" \
    --alpha_stab "$ALPHA_STAB" --beta_switch "$BETA_SWITCH" \
    --gamma_coh_floor "$GAMMA_FLOOR" --tau_coh_floor "$TAU_FLOOR" &

run_train multi_C_floor_drop 2 \
    --gate_variant multi_C \
    --target_repair_rate "$TARGET_RATE" --gamma_rate "$GAMMA_RATE" \
    --alpha_stab "$ALPHA_STAB" --beta_switch "$BETA_SWITCH" \
    --gamma_coh_floor "$GAMMA_FLOOR" --tau_coh_floor "$TAU_FLOOR" \
    --gamma_coh_drop "$GAMMA_DROP" --epsilon_coh_drop "$EPSILON_DROP" &

wait
echo "[wave 2 done]"

# ----------------------------- Step 2: eval ----------------------------------
echo ""
echo "===== STEP 2: eval all 6 models + 2 baselines ====="

for r in fixed dcp pure_corr multi_C multi_C_floor multi_C_floor_drop; do
    OUT_DIR="$OUT_BASE/eval_run_${r}_v6"
    if [ -f "$OUT_DIR/metrics.json" ]; then
        echo "[skip eval] $r"
        continue
    fi
    echo "[eval] $r"
    python -m src.eval.eval_tom_model \
        --checkpoint "$OUT_BASE/run_${r}_v6/model.pt" \
        --split_manifest "$SPLIT_MANIFEST" \
        --image_map_jsonl "$IMAGE_MAP" \
        --out_dir "$OUT_DIR"
done

# Baselines (deterministic — no need to redo per seed, but cheap so we do anyway)
for mode in clip random; do
    OUT_DIR="$OUT_BASE/baseline_${mode}_v6"
    if [ -f "$OUT_DIR/metrics.json" ]; then
        echo "[skip baseline] $mode"
        continue
    fi
    echo "[baseline] $mode"
    python -m src.eval.baselines --mode "$mode" \
        --split_manifest "$SPLIT_MANIFEST" \
        --image_map_jsonl "$IMAGE_MAP" \
        --out_dir "$OUT_DIR"
done

# ----------------------------- Step 3: per-seed main table -------------------
echo ""
echo "===== STEP 3: per-seed main table ====="
python -m src.eval.compare_runs \
    --runs "$OUT_BASE/baseline_random_v6" \
           "$OUT_BASE/baseline_clip_v6" \
           "$OUT_BASE/eval_run_fixed_v6" \
           "$OUT_BASE/eval_run_dcp_v6" \
           "$OUT_BASE/eval_run_multi_C_v6" \
           "$OUT_BASE/eval_run_multi_C_floor_v6" \
           "$OUT_BASE/eval_run_multi_C_floor_drop_v6" \
           "$OUT_BASE/eval_run_pure_corr_v6" \
    --names random clip_only fixed dcp_B multi_C multi_C+floor multi_C+floor+drop pure_corr \
    --out_csv "$OUT_BASE/main_table_v6.csv"

# ----------------------------- Step 4: modality ablation ---------------------
echo ""
echo "===== STEP 4: modality ablation on winner (multi_C+floor) ====="
WINNER_CK="$OUT_BASE/run_multi_C_floor_v6/model.pt"
ABLATE_BASE="$OUT_BASE/ablate_v6"

run_ablate() {
    local NAME="$1"; shift
    local OUT_DIR="$ABLATE_BASE/$NAME"
    if [ -f "$OUT_DIR/metrics.json" ]; then
        echo "[skip ablate] $NAME"
        return
    fi
    echo "[ablate] $NAME"
    python -m src.eval.eval_tom_model \
        --checkpoint "$WINNER_CK" \
        --split_manifest "$SPLIT_MANIFEST" \
        --image_map_jsonl "$IMAGE_MAP" \
        --out_dir "$OUT_DIR" \
        "$@"
}

run_ablate full
run_ablate img_zero   --ablate_image zero
run_ablate img_random --ablate_image random
run_ablate img_mean   --ablate_image mean
run_ablate txt_zero   --ablate_text  zero
run_ablate txt_random --ablate_text  random
run_ablate txt_mean   --ablate_text  mean

# ----------------------------- Step 5: per-seed ablation table ---------------
echo ""
echo "===== STEP 5: per-seed ablation table ====="
python -m src.eval.compare_runs \
    --runs "$ABLATE_BASE/full" \
           "$ABLATE_BASE/img_zero"   "$ABLATE_BASE/img_random"   "$ABLATE_BASE/img_mean" \
           "$ABLATE_BASE/txt_zero"   "$ABLATE_BASE/txt_random"   "$ABLATE_BASE/txt_mean" \
    --names full img_zero img_random img_mean txt_zero txt_random txt_mean \
    --out_csv "$OUT_BASE/ablation_table_v6.csv"

# ----------------------------- Done ------------------------------------------
echo ""
echo "===================================================================="
echo " SEED $SEED DONE"
echo "  per-seed main table:     $OUT_BASE/main_table_v6.csv"
echo "  per-seed ablation table: $OUT_BASE/ablation_table_v6.csv"
echo "===================================================================="

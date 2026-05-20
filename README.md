# Continuous Belief Modulation for Visually Grounded Dialogue

This repository contains the code for the PhotoChat experiments in a simulation framework for turn-by-turn preserve/revise decisions in visually grounded dialogue.

When a listener receives utterances about a scene it cannot directly observe, it must decide whether to preserve its current grounding state or revise it as new language arrives. This project compares several update rules for that decision and evaluates them through image retrieval performance and intrinsic measures of grounding dynamics.

## Update Rules

The main training pipeline compares:

- `fixed`: fixed-rate updating.
- `dcp_B`: mismatch-driven revision based on KL change-point signals.
- `multi_C`: uncertainty-sensitive revision using mismatch, previous alignment, predicted alignment change, and evidence entropy.
- `multi_C+floor`: uncertainty-sensitive revision with a coherence floor regularizer.
- `multi_C+floor+drop`: uncertainty-sensitive revision with coherence floor and drop regularizers.
- `pure_corr`: retrieval-only objective without stability/switching regularization.

The implementation uses frozen CLIP features, a trainable alignment head, and sequential evidence dynamics over dialogue turns.

## Repository Structure

```text
config/
  requirements.txt        # pinned pip dependencies
  environment.yml         # conda/mamba environment

src/
  dataloaders/            # PhotoChat loader and deterministic shard split
  models/                 # CLIP wrapper, alignment head, gates, losses
  train/                  # training entry point
  eval/                   # retrieval eval, baselines, ablations, analyses
  run_v6.sh               # full single-seed pipeline
  run_all_seeds.sh        # multi-seed pipeline and aggregation
```

## Environment

Python 3.11 is recommended. Install the pinned dependencies with pip:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r config/requirements.txt
```

Or create the environment with conda/mamba:

```bash
mamba env create -f config/environment.yml
conda activate tag-photochat
```

If you use a different CUDA version, install the matching PyTorch build first, then install the remaining packages from `config/requirements.txt`.

## Data

The code expects the PhotoChat data under `data/photochat`:

```text
data/photochat/
  images/
  train_json/
    train_00.json
    ...
    train_20.json
  train_image_photo_desc.jsonl
```

If the dataset is stored elsewhere, create a symlink:

```bash
mkdir -p data
ln -s /path/to/photochat data/photochat
```

Check the expected files:

```bash
ls data/photochat/train_json
ls data/photochat/train_image_photo_desc.jsonl
```

## Smoke Test

Create a deterministic split manifest:

```bash
python -m src.dataloaders.make_split \
  --shards_dir data/photochat/train_json \
  --n_test 2 \
  --seed 42 \
  --out outputs/split_v1.json
```

Run a small training job:

```bash
python -m src.train.train_mvp_wa_wh \
  --split_manifest outputs/split_v1.json \
  --image_map_jsonl data/photochat/train_image_photo_desc.jsonl \
  --turn_window preveal \
  --gate_variant dcp_B \
  --epochs 1 \
  --batch_size 2 \
  --max_items 8 \
  --out_dir outputs/smoke_dcp
```

Evaluate the smoke-test checkpoint:

```bash
python -m src.eval.eval_tom_model \
  --checkpoint outputs/smoke_dcp/model.pt \
  --split_manifest outputs/split_v1.json \
  --image_map_jsonl data/photochat/train_image_photo_desc.jsonl \
  --turn_window preveal \
  --max_dialogues 8 \
  --out_dir outputs/smoke_eval_dcp
```

Run the two retrieval baselines:

```bash
python -m src.eval.baselines \
  --mode clip \
  --split_manifest outputs/split_v1.json \
  --image_map_jsonl data/photochat/train_image_photo_desc.jsonl \
  --turn_window preveal \
  --out_dir outputs/smoke_baseline_clip

python -m src.eval.baselines \
  --mode random \
  --split_manifest outputs/split_v1.json \
  --image_map_jsonl data/photochat/train_image_photo_desc.jsonl \
  --turn_window preveal \
  --out_dir outputs/smoke_baseline_random
```

## Reproducing Main Runs

Run the full single-seed pipeline:

```bash
SEED=0 TURN_WINDOW=preveal bash src/run_v6.sh
```

This script:

1. Creates or reuses `outputs/split_v1.json`.
2. Trains all update-rule variants.
3. Evaluates all trained models.
4. Runs CLIP-only and random baselines.
5. Runs modality ablations.
6. Writes per-seed result tables.

Outputs for `TURN_WINDOW=preveal` are written under:

```text
outputs/seed_0_preveal/
```

For full-dialogue runs:

```bash
SEED=0 TURN_WINDOW=full bash src/run_v6.sh
```

Outputs are then written under:

```text
outputs/seed_0/
```

## Multi-Seed Runs

Run seeds 0, 1, and 2, then aggregate mean and standard deviation tables:

```bash
TURN_WINDOW=preveal SEEDS="0 1 2" bash src/run_all_seeds.sh
```

The aggregate tables are:

```text
outputs/main_table_meanstd_preveal.csv
outputs/ablation_table_meanstd_preveal.csv
```

For full-dialogue results:

```bash
TURN_WINDOW=full SEEDS="0 1 2" bash src/run_all_seeds.sh
```

## Post-Hoc Analyses

Run stratified retrieval analysis for one seed:

```bash
python -m src.eval.eval_stratified \
  --seed_dir outputs/seed_0_preveal \
  --split_manifest outputs/split_v1.json \
  --image_map_jsonl data/photochat/train_image_photo_desc.jsonl \
  --turn_window preveal \
  --out_dir outputs/seed_0_preveal/stratified_v6
```

Summarize stratified results across seeds:

```bash
python -m src.eval.summarize_stratified_seeds \
  --seed_dirs outputs/seed_0_preveal outputs/seed_1_preveal outputs/seed_2_preveal \
  --out_dir outputs
```

Analyze the learned multi-signal gate:

```bash
python -m src.eval.analyze_features \
  --checkpoint outputs/seed_0_preveal/run_multi_C_v6/model.pt \
  --split_manifest outputs/split_v1.json \
  --image_map_jsonl data/photochat/train_image_photo_desc.jsonl \
  --max_dialogues 200 \
  --out_dir outputs/viz/feature_seed_0
```

Plot R@K curves:

```bash
python -m src.eval.plot_rk_curve \
  --runs "outputs/seed_*_preveal/baseline_clip_v6" \
         "outputs/seed_*_preveal/eval_run_dcp_v6" \
         "outputs/seed_*_preveal/eval_run_multi_C_v6" \
         "outputs/seed_*_preveal/eval_run_multi_C_floor_v6" \
  --names clip_only dcp_B multi_C multi_C+floor \
  --out outputs/viz/rk_curve_meanstd.png
```

Visualize per-turn revision dynamics:

```bash
python -m src.eval.viz_dynamics hist \
  --runs outputs/seed_0_preveal/eval_run_multi_C_floor_v6 \
  --out outputs/viz/rho_hist.png
```

## Turn Windows

Two dialogue windows are supported:

- `TURN_WINDOW=preveal`: use turns before the first `share_photo=True` event.
- `TURN_WINDOW=full`: use the full dialogue.

The paper's listener-before-image setting corresponds to `preveal`.

## Outputs

Common output files include:

```text
model.pt                         # trained checkpoint
loss.csv                         # training losses
metrics.json                     # retrieval and dynamics metrics
per_turn.csv                     # per-turn rho/alignment/KL traces
ranks.npy                        # retrieval ranks for R@K plots
main_table_v6.csv                # per-seed main comparison
ablation_table_v6.csv            # per-seed ablation comparison
```

Large outputs, checkpoints, local data symlinks, and virtual environments should not be committed.

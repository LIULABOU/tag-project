# TAG Project (PhotoChat)

This repo contains:
- Sanity visualization for PhotoChat (alignment trajectory + patch heatmaps)
- Training script for a CLIP-style contrastive baseline (loss curve)

## 1) Server Setup

### 1.1 Clone the repository
```bash
cd /home/liu0193/Data/Bhanu
git clone https://github.com/jbhanuchai/tag-project.git
cd tag-project
```
If the repo already exists:
```bash
git pull
```

### Link the PhotoChat dataset into the repo

On the server, the dataset is located at: 
```bash
/home/liu0193/Data/Bhanu/photochat
```

Create a symlink so the code can access it as data/photochat/...
```bash
mkdir -p data
ln -s /home/liu0193/Data/Bhanu/photochat data/photochat
```

Verify:
```bash
ls data/photochat
```

Expected structure:
```bash
data/photochat/
    images/
    train_json/
    train_image_photo_desc.jsonl
```

### Create and activate a Python environment

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
## 2) Run Sanity Visualization

This generates:
alignment_trajectory.png
per-turn heatmaps turn_XX_heatmap.png
summary.csv

Run:
```bash
TOKENIZERS_PARALLELISM=false PYTHONPATH=. \
python src/eval/run_sanity_viz.py --num_examples 2
```

Outputs are saved in:

```bash
outputs/sanity/
```

## 3) Train Contrastive Baseline (Loss Curve)
This trains small projection heads (CLIP frozen) with a contrastive (InfoNCE) loss and saves:

Run:
```bash
TOKENIZERS_PARALLELISM=false PYTHONPATH=. \
python src/train/train_full_clip.py --batch_size 16 --epochs 10
```
Outputs:
```bash
outputs/train_full_clip/loss.csv
outputs/train_full_clip/loss.png
```

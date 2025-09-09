#!/usr/bin/env bash
set -e

echo "=== Create and activate virtualenv ==="
python -m venv venv_dpfont
source venv_dpfont/bin/activate

echo "=== Upgrade pip and install requirements ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Quick training run (toy) ==="
# Make sure you have data/images and data/stroke_order prepared as described in README.md
python src/train.py --data_root data --stroke_root data/stroke_order --fonts FontA,FontB --chars 20013 --epochs 1 --batch 8 --out out

echo "=== Sampling example (after checkpoint is saved) ==="
# Adjust checkpoint path as necessary. The sample helper is a function; here we call it via Python -c.
CKPT=out/ckpt_epoch_0.pt
python - <<PY
from src.sample import run_sample
# single example: content codepoint 20013, stroke list as zeros (36,), style id 0
run_sample(CKPT, [20013], [[0]*36], [0], out_dir='samples')
PY

echo "Done. Samples saved in ./samples"

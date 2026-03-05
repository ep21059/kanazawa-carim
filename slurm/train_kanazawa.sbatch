#!/bin/bash
#SBATCH --job-name=TrainKanazawa
#SBATCH --output=/home/ryoc1220/carim_ver1/kanazawa_ver/runs/train_%j.out
#SBATCH --error=/home/ryoc1220/carim_ver1/kanazawa_ver/runs/train_%j.err
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error

set -e

singularity exec --nv \
  --bind /home/ryoc1220/carim_ver1:/workspace \
  /home/ryoc1220/carim_ver1/kanazawa_ver/carim_qwen.sif \
  bash -c '
  export PYTHONPATH=/workspace/kanazawa_ver:$PYTHONPATH
  cd /workspace/kanazawa_ver
  
  echo "Starting Kanazawa Fine-Tuning..."
  python3 train.py \
    --jsonl_path datasets/kanazawa_scene/processed/train_full.jsonl \
    --elements_path datasets/kanazawa_scene/captions_elements.json \
    --epochs 5 \
    --batch_size 24 \
    --lr 5e-5 \
    --pretrained /workspace/runs/carim_text_model_full.pt \
    --save_path runs/carim_kanazawa_finetuned.pt
'

#!/bin/bash
#SBATCH --job-name=IdxKanaz
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/ryoc1220/carim_ver1/kanazawa_ver/runs/index_kanazawa_%j.out
#SBATCH --error=/home/ryoc1220/carim_ver1/kanazawa_ver/runs/index_kanazawa_%j.err

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error

set -e

singularity exec --nv \
  --bind /home/ryoc1220/carim_ver1:/workspace \
  /home/ryoc1220/carim_ver1/kanazawa_ver/carim_qwen.sif \
  bash -c '
  export PYTHONPATH=/workspace:$PYTHONPATH
  cd /workspace/kanazawa_ver
  
  echo "Generating Index for Kanazawa dataset..."
  python3 scripts/indexer.py \
    --captions_file datasets/kanazawa_scene/captions_elements.json \
    --output_file datasets/kanazawa_scene/processed/text_index.pt \
    --checkpoint runs/carim_kanazawa_finetuned.pt
'

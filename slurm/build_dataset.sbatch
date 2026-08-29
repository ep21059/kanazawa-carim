#!/bin/bash
#SBATCH --job-name=BuildKanazawa
#SBATCH --output=/home/ryoc1220/work/kanazawa-carim/runs/build_%j.out
#SBATCH --error=/home/ryoc1220/work/kanazawa-carim/runs/build_%j.err
#SBATCH --partition=a6000_ada
#SBATCH --time=00:10:00
#SBATCH --mem=8G

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error

set -e

singularity exec --nv \
  --bind /home/ryoc1220/projects/carim:/workspace \
  /home/ryoc1220/containers/carim_qwen.sif \
  bash -c '
  export PYTHONPATH=/workspace/kanazawa_ver:$PYTHONPATH
  cd /workspace/kanazawa_ver
  
  echo "Building initial Kanazawa JSONL..."
  python3 scripts/build_dataset_from_images.py
'

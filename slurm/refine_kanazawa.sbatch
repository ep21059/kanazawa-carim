#!/bin/bash
#SBATCH --job-name=RefineKanazawa
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ryoc1220/carim_ver1/kanazawa_ver/runs/refine_kanazawa_%j.out
#SBATCH --error=/home/ryoc1220/carim_ver1/kanazawa_ver/runs/refine_kanazawa_%j.err

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
  
  echo "Refining Kanazawa Captions..."
  python3 scripts/refine_captions_llm.py \
    --json_path datasets/kanazawa_scene/captions_inclusive.json \
    --output_file datasets/kanazawa_scene/captions_elements.json \
    --model_path Qwen/Qwen2-1.5B-Instruct
'

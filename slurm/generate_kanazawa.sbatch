#!/bin/bash
#SBATCH --job-name=GenKanazawa
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/ryoc1220/work/kanazawa-carim/runs/gen_kanazawa_%j.out
#SBATCH --error=/home/ryoc1220/work/kanazawa-carim/runs/gen_kanazawa_%j.err

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
  
  echo "Generating Captions for Kanazawa dataset..."
  python3 scripts/generate_captions_qwen.py \
    --data_dir datasets/kanazawa_scene/samples/CAM_FRONT \
    --output_file datasets/kanazawa_scene/captions_inclusive.json \
    --model_path Qwen/Qwen-VL-Chat \
    --batch_size 1
'

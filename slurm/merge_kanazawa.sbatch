#!/bin/bash
#SBATCH --job-name=MergeKanazawa
#SBATCH --output=/home/ryoc1220/carim_ver1/kanazawa_ver/runs/merge_%j.out
#SBATCH --error=/home/ryoc1220/carim_ver1/kanazawa_ver/runs/merge_%j.err
#SBATCH --partition=a6000_ada
#SBATCH --time=00:10:00
#SBATCH --mem=8G

#SLACK: notify-start
#SLACK: notify-end
#SLACK: notify-error

cd /home/ryoc1220/carim_ver1/kanazawa_ver
export PYTHONPATH=/home/ryoc1220/carim_ver1/kanazawa_ver:$PYTHONPATH

python3 scripts/merge_full_dataset.py \
    --train_jsonl datasets/kanazawa_scene/processed/train.jsonl \
    --captions_json datasets/kanazawa_scene/captions_inclusive.json \
    --elements_json datasets/kanazawa_scene/captions_elements.json \
    --output_jsonl datasets/kanazawa_scene/processed/train_full.jsonl

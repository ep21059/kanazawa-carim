#!/bin/bash
#SBATCH --job-name=AppKanaz
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/ryoc1220/carim_ver1/kanazawa_ver/runs/app_kanazawa_%j.out
#SBATCH --error=/home/ryoc1220/carim_ver1/kanazawa_ver/runs/app_kanazawa_%j.err

set -e

# Port configuration (Streamlit defaults to 8501, using a standard fallback just in case)
PORT=8501

echo "Starting CARIM Search App (Kanazawa ver) on port ${PORT}..."
echo "To access, create an SSH tunnel from your local machine:"
echo "ssh -N -f -L ${PORT}:$(hostname):${PORT} ryoc1220@<YOUR_CLUSTER_IP>"

singularity exec --nv \
  --bind /home/ryoc1220/carim_ver1:/workspace \
  /home/ryoc1220/carim_ver1/kanazawa_ver/carim_qwen.sif \
  bash -c "
  export PYTHONPATH=/workspace:\$PYTHONPATH
  cd /workspace/kanazawa_ver
  
  streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0
"

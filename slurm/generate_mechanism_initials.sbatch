#!/usr/bin/bash
#SBATCH --job-name=mech-matrix-inputs
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --gres=gpu:nvlink4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/baix2/wan-i2v-state/runs/logs/%x-%j.out
#SBATCH --error=/home/baix2/wan-i2v-state/runs/logs/%x-%j.err

set -euo pipefail
repo_root=/home/baix2/wan-i2v-physics-mechanisms
state_root=/home/baix2/wan-i2v-state
env_dir="${state_root}/envs/mechanism-matrix-v1"
manifest="${repo_root}/configs/mechanism_matrix_v1.json"
data_root="${state_root}/data/mechanism-matrix-v1"

module purge
module load standard
module load gcc/13
module load python/3.12
module load cuda/12.8
export HF_HOME="${state_root}/cache/huggingface"
export TOKENIZERS_PARALLELISM=false

"${env_dir}/bin/python" "${repo_root}/scripts/prepare_mechanism_matrix.py" generate-phygen \
  --manifest "${manifest}" \
  --data-root "${data_root}" \
  --image-model "${state_root}/models/sdxl-turbo"
"${env_dir}/bin/python" "${repo_root}/scripts/prepare_mechanism_matrix.py" lock \
  --manifest "${manifest}" \
  --data-root "${data_root}" \
  --lock-output "${data_root}/input_lock.json"

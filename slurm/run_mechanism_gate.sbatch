#!/usr/bin/bash
#SBATCH --job-name=mech-matrix-gate
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --gres=gpu:nvlink4:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/home/baix2/wan-i2v-state/runs/logs/%x-%j.out
#SBATCH --error=/home/baix2/wan-i2v-state/runs/logs/%x-%j.err

set -euo pipefail
: "${SHARD:?submit with --export=ALL,SHARD=0 or SHARD=1}"
if [[ "${SHARD}" == 0 ]]; then
  model_keys=(wan21_i2v)
elif [[ "${SHARD}" == 1 ]]; then
  model_keys=(cogvideox_i2v ltx_video_i2v)
else
  echo "SHARD must be 0 or 1" >&2
  exit 2
fi

repo_root=/home/baix2/wan-i2v-physics-mechanisms
state_root=/home/baix2/wan-i2v-state
env_dir="${state_root}/envs/mechanism-matrix-v1"
manifest="${repo_root}/configs/mechanism_matrix_v1.json"
output_root="${state_root}/runs/mechanism-matrix-v1"
module purge
module load standard
module load gcc/13
module load python/3.12
module load cuda/12.8
export HF_HOME="${state_root}/cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export STUDY_REVISION="${STUDY_REVISION:-unknown}"

for model_key in "${model_keys[@]}"; do
  "${env_dir}/bin/python" "${repo_root}/scripts/run_mechanism_matrix.py" \
    --manifest "${manifest}" \
    --model-key "${model_key}" \
    --case-id physics_iq_pair_balls_collide \
    --seed 42 \
    --arm baseline --arm phaselock \
    --models-root "${state_root}/models" \
    --phaselock-code "${state_root}/upstream/phaselock/code" \
    --data-root "${state_root}/data/mechanism-matrix-v1" \
    --input-lock "${state_root}/data/mechanism-matrix-v1/input_lock.json" \
    --output-root "${output_root}"
done

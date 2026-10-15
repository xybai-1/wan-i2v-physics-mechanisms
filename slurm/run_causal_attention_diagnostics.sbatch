#!/usr/bin/bash
#SBATCH --job-name=causal-attn-diag
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --gres=gpu:nvlink4:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --output=/home/baix2/wan-i2v-state/runs/logs/%x-%j.out
#SBATCH --error=/home/baix2/wan-i2v-state/runs/logs/%x-%j.err

set -euo pipefail
: "${SHARD:?submit with --export=ALL,SHARD=0 or SHARD=1}"
: "${MODE:?submit with --export=ALL,MODE=smoke or MODE=full}"
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
diagnostic="${repo_root}/configs/causal_attention_diagnostics_v1.json"
if [[ "${MODE}" == smoke ]]; then
  output_root="${state_root}/runs/causal-attention-diagnostics-v1-smoke"
  case_ids=(phyground_pair_balls_collide)
  arm_args=(--arm baseline)
elif [[ "${MODE}" == full ]]; then
  output_root="${state_root}/runs/causal-attention-diagnostics-v1"
  mapfile -t case_ids < <("${env_dir}/bin/python" - "${diagnostic}" <<'PY'
import json, sys
for item in json.load(open(sys.argv[1], encoding="utf-8"))["cases"]:
    print(item["case_id"])
PY
  )
  arm_args=(--arm baseline --arm phaselock)
else
  echo "MODE must be smoke or full" >&2
  exit 2
fi

module purge
module load standard
module load gcc/13
module load python/3.12
module load cuda/12.8
export HF_HOME="${state_root}/cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export STUDY_REVISION="${STUDY_REVISION:-unknown}"

case_args=()
for case_id in "${case_ids[@]}"; do case_args+=(--case-id "${case_id}"); done
for model_key in "${model_keys[@]}"; do
  "${env_dir}/bin/python" "${repo_root}/scripts/run_causal_attention_diagnostic.py" \
    --manifest "${manifest}" \
    --diagnostic-config "${diagnostic}" \
    --model-key "${model_key}" \
    "${case_args[@]}" \
    --seed 42 \
    "${arm_args[@]}" \
    --models-root "${state_root}/models" \
    --phaselock-code "${state_root}/upstream/phaselock/code" \
    --data-root "${state_root}/data/mechanism-matrix-v1" \
    --input-lock "${state_root}/data/mechanism-matrix-v1/input_lock_flux.json" \
    --reference-root "${state_root}/runs/mechanism-matrix-v1" \
    --output-root "${output_root}"
done

#!/usr/bin/bash
#SBATCH --job-name=causal-route
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --gres=gpu:nvlink4:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/home/baix2/wan-i2v-state/runs/logs/%x-%j.out
#SBATCH --error=/home/baix2/wan-i2v-state/runs/logs/%x-%j.err

set -euo pipefail
: "${MODE:?submit with --export=ALL,MODE=smoke or MODE=full}"
: "${SHARD:=0}"

repo_root=/home/baix2/wan-i2v-physics-mechanisms
state_root=/home/baix2/wan-i2v-state
env_dir="${state_root}/envs/mechanism-matrix-v1"
manifest="${repo_root}/configs/mechanism_matrix_v1.json"
diagnostic="${repo_root}/configs/causal_attention_diagnostics_v1.json"
intervention="${repo_root}/configs/causal_routing_intervention_v1.json"

if [[ "${MODE}" == smoke ]]; then
  output_root="${state_root}/runs/causal-routing-intervention-v1-smoke"
  case_ids=(phygenbench_chain_fire_paper)
elif [[ "${MODE}" == full && "${SHARD}" == 0 ]]; then
  output_root="${state_root}/runs/causal-routing-intervention-v1"
  case_ids=(phyground_pair_balls_collide phyground_chain_domino_gap)
elif [[ "${MODE}" == full && "${SHARD}" == 1 ]]; then
  output_root="${state_root}/runs/causal-routing-intervention-v1"
  case_ids=(phygenbench_chain_fire_paper physics_iq_single_ball_free)
else
  echo "MODE must be smoke, or MODE=full with SHARD=0/1" >&2
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

"${env_dir}/bin/python" "${repo_root}/scripts/run_causal_routing_intervention.py" \
  --manifest "${manifest}" \
  --diagnostic-config "${diagnostic}" \
  --intervention-config "${intervention}" \
  "${case_args[@]}" \
  --variant forward \
  --variant reverse_time \
  --variant same_time \
  --models-root "${state_root}/models" \
  --phaselock-code "${state_root}/upstream/phaselock/code" \
  --data-root "${state_root}/data/mechanism-matrix-v1" \
  --input-lock "${state_root}/data/mechanism-matrix-v1/input_lock_flux.json" \
  --reference-root "${state_root}/runs/mechanism-matrix-v1" \
  --output-root "${output_root}"

#!/usr/bin/bash
#SBATCH --job-name=wan-attn-8
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
if [[ "${SHARD}" != 0 && "${SHARD}" != 1 ]]; then
  echo "SHARD must be 0 or 1" >&2
  exit 2
fi

repo_root=/home/baix2/wan-i2v-physics-mechanisms
state_root=/home/baix2/wan-i2v-state
manifest="${repo_root}/configs/phyground_attention_8.json"
module purge
module load standard
module load gcc/13
module load python/3.12
module load cuda/12.8
export HF_HOME="${state_root}/cache/huggingface"
export TOKENIZERS_PARALLELISM=false

mapfile -t case_ids < <(
  "${state_root}/envs/wan-smoke/bin/python" - "${manifest}" "${SHARD}" <<'PY'
import json
import sys
manifest, shard = sys.argv[1], int(sys.argv[2])
cases = json.load(open(manifest, encoding="utf-8"))["cases"]
for index, case in enumerate(cases):
    if index % 2 == shard:
        print(case["video_id"])
PY
)

for case_id in "${case_ids[@]}"; do
  for arm in baseline phaselock; do
    run_dir="${state_root}/runs/attention8/${case_id}/seed_42/${arm}"
    if [[ -f "${run_dir}/run.json" ]]; then
      echo "already complete: ${case_id} ${arm}"
      continue
    fi
    if [[ -e "${run_dir}" ]]; then
      echo "partial output requires inspection; refusing overwrite: ${run_dir}" >&2
      exit 3
    fi
    echo "starting ${case_id} ${arm}"
    "${state_root}/envs/wan-smoke/bin/python" "${repo_root}/scripts/run_attention_probe.py" \
      --arm "${arm}" \
      --manifest "${manifest}" \
      --case-id "${case_id}" \
      --seed 42 \
      --model "${state_root}/models/Wan2.1-I2V-14B-720P-Diffusers" \
      --phaselock-code "${state_root}/upstream/phaselock/code" \
      --data-root "${state_root}/data/phyground" \
      --output-root "${state_root}/runs/attention8" \
      --reference-root "${state_root}/runs/reference-unavailable"
  done
done

#!/usr/bin/bash
#SBATCH --job-name=wan-attn-probe
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --gres=gpu:nvlink4:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=1-00:00
#SBATCH --output=/home/baix2/wan-i2v-state/runs/logs/%x-%j.out
#SBATCH --error=/home/baix2/wan-i2v-state/runs/logs/%x-%j.err

set -euo pipefail
: "${ARM:?submit with --export=ALL,ARM=baseline or ARM=phaselock}"

repo_root=/home/baix2/wan-i2v-physics-mechanisms
state_root=/home/baix2/wan-i2v-state
module purge
module load standard
module load gcc/13
module load python/3.12
module load cuda/12.8
export HF_HOME="${state_root}/cache/huggingface"
export TOKENIZERS_PARALLELISM=false

"${state_root}/envs/wan-smoke/bin/python" "${repo_root}/scripts/run_attention_probe.py" \
  --arm "${ARM}" \
  --manifest "${repo_root}/configs/smoke_mechanics_411.json" \
  --case-id mechanics_411 \
  --seed 42 \
  --model "${state_root}/models/Wan2.1-I2V-14B-720P-Diffusers" \
  --phaselock-code "${state_root}/upstream/phaselock/code" \
  --data-root "${state_root}/data/phyground" \
  --output-root "${state_root}/runs/attention" \
  --reference-root "${state_root}/runs/smoke"

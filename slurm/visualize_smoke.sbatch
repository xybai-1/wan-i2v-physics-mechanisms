#!/usr/bin/bash
#SBATCH --job-name=wan-mech-viz
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00
#SBATCH --output=/home/baix2/wan-i2v-state/runs/logs/%x-%j.out
#SBATCH --error=/home/baix2/wan-i2v-state/runs/logs/%x-%j.err

set -euo pipefail

repo_root=/home/baix2/wan-i2v-physics-mechanisms
state_root=/home/baix2/wan-i2v-state
base="${state_root}/runs/smoke/mechanics_411/seed_42"

module purge
module load standard python/3.12

"${state_root}/envs/wan-smoke/bin/python" "${repo_root}/scripts/visualize_smoke.py" \
  --baseline-dir "${base}/baseline" \
  --phaselock-dir "${base}/phaselock" \
  --output-dir "${base}/analysis"

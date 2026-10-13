#!/usr/bin/bash
#SBATCH --job-name=wan-mech-env
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --gres=gpu:nvlink4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00
#SBATCH --output=/home/baix2/wan-i2v-state/runs/logs/%x-%j.out
#SBATCH --error=/home/baix2/wan-i2v-state/runs/logs/%x-%j.err

set -euo pipefail
mkdir -p /home/baix2/wan-i2v-state/runs/logs

REPO_ROOT=/home/baix2/wan-i2v-physics-mechanisms \
STATE_ROOT=/home/baix2/wan-i2v-state \
  bash /home/baix2/wan-i2v-physics-mechanisms/scripts/setup_generation_env.sh

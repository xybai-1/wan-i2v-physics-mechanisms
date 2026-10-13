#!/usr/bin/bash
#SBATCH --job-name=wan-mech-setup
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00
#SBATCH --output=/home/baix2/wan-i2v-state/runs/logs/%x-%j.out
#SBATCH --error=/home/baix2/wan-i2v-state/runs/logs/%x-%j.err

set -euo pipefail

module purge
module load standard

mkdir -p /home/baix2/wan-i2v-state/runs/logs

REPO_ROOT=/home/baix2/wan-i2v-physics-mechanisms \
STATE_ROOT=/home/baix2/wan-i2v-state \
  bash /home/baix2/wan-i2v-physics-mechanisms/scripts/bootstrap_goose.sh


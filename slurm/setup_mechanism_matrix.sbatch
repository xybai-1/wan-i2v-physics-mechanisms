#!/usr/bin/bash
#SBATCH --job-name=mech-matrix-setup
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/baix2/wan-i2v-state/runs/logs/%x-%j.out
#SBATCH --error=/home/baix2/wan-i2v-state/runs/logs/%x-%j.err

set -euo pipefail
mkdir -p /home/baix2/wan-i2v-state/runs/logs
/home/baix2/wan-i2v-physics-mechanisms/scripts/setup_mechanism_matrix.sh

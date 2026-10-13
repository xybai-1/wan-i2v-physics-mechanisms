#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/home/baix2/wan-i2v-physics-mechanisms}"
state_root="${STATE_ROOT:-/home/baix2/wan-i2v-state}"
env_dir="${state_root}/envs/wan-smoke"
model_dir="${state_root}/models/Wan2.1-I2V-14B-720P-Diffusers"
model_revision="eb849f76dfa246545b65774a9e25943ee69b3fa3"
phaselock_dir="${state_root}/upstream/phaselock"

module purge
module load standard
module load gcc/13
module load python/3.12
module load cuda/12.8

if [[ ! -x "${env_dir}/bin/python" ]]; then
  python -m venv "${env_dir}"
fi

export HF_HOME="${state_root}/cache/huggingface"
export PIP_CACHE_DIR="${state_root}/cache/pip"
mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}" "${model_dir}"

"${env_dir}/bin/python" -m pip install --upgrade pip wheel setuptools
"${env_dir}/bin/python" -m pip install \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu126
"${env_dir}/bin/python" -m pip install \
  diffusers==0.35.2 transformers==4.49.0 accelerate==1.10.1 \
  sentencepiece protobuf safetensors hf_xet ftfy \
  "huggingface_hub[hf_transfer]" "imageio[ffmpeg]" pillow numpy matplotlib
"${env_dir}/bin/python" -m pip install -e "${phaselock_dir}/code"
"${env_dir}/bin/python" -m pip check

MODEL_DIR="${model_dir}" MODEL_REVISION="${model_revision}" \
  "${env_dir}/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    revision=os.environ["MODEL_REVISION"],
    local_dir=os.environ["MODEL_DIR"],
)
PY

"${env_dir}/bin/python" "${repo_root}/scripts/fetch_smoke_inputs.py" \
  "${repo_root}/configs/smoke_mechanics_411.json" \
  --data-root "${state_root}/data/phyground"

"${env_dir}/bin/python" - <<'PY'
import torch
from diffusers import WanImageToVideoPipeline
from phaselock.guidance import LatentDeltaGuidance
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("pipeline", WanImageToVideoPipeline.__name__)
print("phaselock", LatentDeltaGuidance.__name__)
PY

#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/home/baix2/wan-i2v-physics-mechanisms}"
state_root="${STATE_ROOT:-/home/baix2/wan-i2v-state}"
env_dir="${state_root}/envs/mechanism-matrix-v1"
models_root="${state_root}/models"
phaselock_dir="${state_root}/upstream/phaselock"
manifest="${repo_root}/configs/mechanism_matrix_v1.json"

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
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}" "${models_root}" "${state_root}/data/mechanism-matrix-v1"

"${env_dir}/bin/python" -m pip install --upgrade pip wheel setuptools
"${env_dir}/bin/python" -m pip install \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu126
"${env_dir}/bin/python" -m pip install \
  diffusers==0.35.2 transformers==4.49.0 accelerate==1.10.1 \
  sentencepiece protobuf safetensors hf_xet ftfy \
  "huggingface_hub[hf_transfer]" "imageio[ffmpeg]" pillow numpy
"${env_dir}/bin/python" -m pip install -e "${phaselock_dir}/code"
"${env_dir}/bin/python" -m pip check

MANIFEST="${manifest}" MODELS_ROOT="${models_root}" "${env_dir}/bin/python" - <<'PY'
import json
import os
from huggingface_hub import snapshot_download

manifest = json.load(open(os.environ["MANIFEST"], encoding="utf-8"))
root = os.environ["MODELS_ROOT"]
downloads = [
    (item["repo_id"], item["revision"], item["local_dir"])
    for item in manifest["models"].values()
]
downloads.append((
    "stabilityai/sdxl-turbo",
    manifest["source_revisions"]["sdxl_turbo"],
    "sdxl-turbo",
))
for repo_id, revision, local_dir in downloads:
    destination = os.path.join(root, local_dir)
    print(f"snapshot {repo_id}@{revision} -> {destination}", flush=True)
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=destination)
PY

"${env_dir}/bin/python" "${repo_root}/scripts/prepare_mechanism_matrix.py" fetch-public \
  --manifest "${manifest}" \
  --data-root "${state_root}/data/mechanism-matrix-v1"

"${env_dir}/bin/python" - <<'PY'
import torch
from diffusers import AutoPipelineForText2Image, CogVideoXImageToVideoPipeline, LTXImageToVideoPipeline, WanImageToVideoPipeline
from phaselock.guidance import LatentDeltaGuidance
print("torch", torch.__version__, "cuda-build", torch.version.cuda)
print("pipelines", WanImageToVideoPipeline.__name__, CogVideoXImageToVideoPipeline.__name__, LTXImageToVideoPipeline.__name__, AutoPipelineForText2Image.__name__)
print("phaselock", LatentDeltaGuidance.__name__)
PY

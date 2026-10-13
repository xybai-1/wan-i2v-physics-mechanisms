#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/home/baix2/wan-i2v-physics-mechanisms}"
state_root="${STATE_ROOT:-/home/baix2/wan-i2v-state}"
upstream_root="${state_root}/upstream"
study_revision="${STUDY_REVISION:-}"

if [[ -z "${study_revision}" ]] && git -C "${repo_root}" rev-parse HEAD >/dev/null 2>&1; then
  study_revision="$(git -C "${repo_root}" rev-parse HEAD)"
fi
if [[ -z "${study_revision}" ]]; then
  echo "STUDY_REVISION is required when the deployed checkout has no .git directory" >&2
  exit 2
fi

mkdir -p \
  "${state_root}/envs" \
  "${upstream_root}" \
  "${state_root}/models" \
  "${state_root}/data" \
  "${state_root}/cache/huggingface" \
  "${state_root}/cache/pip" \
  "${state_root}/runs" \
  "${state_root}/provenance"

clone_or_update() {
  local name="$1"
  local url="$2"
  local destination="${upstream_root}/${name}"
  if [[ -d "${destination}/.git" ]]; then
    git -C "${destination}" fetch --all --tags --prune
  else
    git clone "${url}" "${destination}"
  fi
  git -C "${destination}" rev-parse HEAD
}

{
  printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'host=%s\n' "$(hostname --fqdn)"
  printf 'study_repo=%s\n' "${study_revision}"
  printf 'alignvid=%s\n' "$(clone_or_update AlignVid https://github.com/LAW1223/AlignVid.git)"
  printf 'physics=%s\n' "$(clone_or_update physics https://github.com/Siriuslala/physics.git)"
  printf 'phaselock=%s\n' "$(clone_or_update phaselock https://github.com/dnwjddl/phaselock.git)"
  printf 'phyground=%s\n' "$(clone_or_update PhyGround https://github.com/NU-World-Model-Embodied-AI/PhyGround.git)"
} | tee "${state_root}/provenance/upstreams.txt"

echo "Upstream source deployment completed. No model weights or GPU jobs were launched."


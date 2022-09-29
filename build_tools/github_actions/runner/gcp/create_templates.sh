#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Creates instance templates for (cpu, gpu) x (presubmit, postsubmit) instances
# according to the current configuration.

set -euo pipefail

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";

TESTING="${TEMPLATE_TESTING:-0}"

TEMPLATE_BASE_NAME="${TEMPLATE_BASE_NAME:-github-runner}"
TEMPLATE_CONFIG_REPO="${TEMPLATE_CONFIG_REPO:-iree-org/iree}"
TEMPLATE_CONFIG_REF="${TEMPLATE_CONFIG_REF:-$(git rev-parse HEAD)}"
GPU_IMAGE="github-runner-gpu-2022-09-29-1664451806"
GPU_DISK_SIZE_GB=100
CPU_IMAGE="github-runner-cpu-2022-09-29-1664451255"
CPU_DISK_SIZE_GB=100

if (( TESTING==0 )); then
  if [[ "${TEMPLATE_CONFIG_REPO}" != iree-org/iree ]]; then
    echo "Expected default settings for non-testing template, but TEMPLATE_CONFIG_REPO='${TEMPLATE_CONFIG_REPO}'. Aborting"
    exit 1
  fi
  if [[ "${TEMPLATE_BASE_NAME}" != github-runner ]]; then
    echo "Expected default settings for non-testing template, but TEMPLATE_BASE_NAME='${TEMPLATE_BASE_NAME}'. Aborting"
    exit 1
  fi
  if [[ "${TEMPLATE_CONFIG_REF}" != "$(git rev-parse main)" ]]; then
    echo "Expected default settings for non-testing template, but TEMPLATE_CONFIG_REF='${TEMPLATE_CONFIG_REF}'. Aborting"
    exit 1
  fi
fi

TIME_STRING="$(date +%Y-%m-%d-%s)"
SHORT_REF="${TEMPLATE_CONFIG_REF:0:10}"
VERSION="${SHORT_REF}-${TIME_STRING}"
STARTUP_SCRIPT_PATH="/tmp/startup_script.${SHORT_REF}.sh"
GITHUB_RUNNER_SCOPE=iree-org
GITHUB_RUNNER_VERSION="2.296.2"
GITHUB_RUNNER_ARCHIVE_DIGEST="34a8f34956cdacd2156d4c658cce8dd54c5aef316a16bbbc95eb3ca4fd76429a"
GITHUB_TOKEN_PROXY_URL="https://ght-proxy-zbhz5clunq-ue.a.run.app"

declare -a METADATA=(
  "github-runner-version=${GITHUB_RUNNER_VERSION}"
  "github-runner-archive-digest=${GITHUB_RUNNER_ARCHIVE_DIGEST}"
  "github-runner-config-ref=${TEMPLATE_CONFIG_REF}"
  "github-runner-config-repo=${TEMPLATE_CONFIG_REPO}"
  "github-runner-scope=${GITHUB_RUNNER_SCOPE}"
  "github-token-proxy-url=${GITHUB_TOKEN_PROXY_URL}"
)

if (( TESTING==1 )); then
  METADATA+=(github-runner-environment=testing)
else
  METADATA+=(github-runner-environment=prod)
fi

declare -a common_args=(
  --project=iree-oss
  # `address=''` indicates an ephemeral IP. This *shouldn't* be necessary here,
  # as the gcloud docs say that this is the default, but in fact if you leave it
  # off the VM gets no external IP and is impossible to SSH into. This knowledge
  # was hard won.
  --network-interface=network=default,address='',network-tier=PREMIUM
  --provisioning-model=STANDARD
  --scopes=https://www.googleapis.com/auth/cloud-platform
  --no-shielded-secure-boot
  --shielded-vtpm
  --shielded-integrity-monitoring
  --reservation-affinity=any
  --metadata-from-file=startup-script="${SCRIPT_DIR}/startup_script.sh"
)


function create_template() {
  local group="$1"
  local type="$2"
  if [[ "${group}" == presubmit ]]; then
    local trust=minimal
  elif [[ "${group}" == postsubmit ]]; then
    local trust=basic
  else
    echo "Got unrecognized group '${group}'" >2
    exit 1
  fi

  local -a metadata=(
    "${METADATA[@]}"
    "github-runner-group=${group}"
    "github-runner-trust=${trust}"
    "github-runner-labels=${type}"
  )

  # Join on commas
  local metadata_string="$(IFS="," ; echo "${metadata[*]}")"

  local -a args=(
    "${TEMPLATE_BASE_NAME}-${group}-${type}-${VERSION}"
    "${common_args[@]}"
    --service-account="github-runner-${trust}-trust@iree-oss.iam.gserviceaccount.com"
    --metadata="${metadata_string}"
  )

  if [[ "${type}" == gpu ]]; then
    args+=(
      --machine-type=a2-highgpu-1g
      --maintenance-policy=TERMINATE
      --accelerator=count=1,type=nvidia-tesla-a100
      --create-disk="auto-delete=yes,boot=yes,image=projects/iree-oss/global/images/${GPU_IMAGE},mode=rw,size=${GPU_DISK_SIZE_GB},type=pd-balanced"
    )
  elif [[ "${type}" == cpu ]]; then
    args+=(
      --machine-type=n1-standard-96
      --maintenance-policy=MIGRATE
      --create-disk="auto-delete=yes,boot=yes,image=projects/iree-oss/global/images/${CPU_IMAGE},mode=rw,size=${CPU_DISK_SIZE_GB},type=pd-balanced"
    )
  else
    echo "Got unrecognized type '${type}'" >2
    exit 1
  fi
  (set -x; gcloud compute instance-templates create "${args[@]}")
  echo
}

for group in presubmit postsubmit; do
  for type in gpu cpu; do
    create_template "${group}" "${type}"
  done
done
echo "Created new templates for version: ${VERSION}"

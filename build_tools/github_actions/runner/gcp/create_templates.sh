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
DRY_RUN="${DRY_RUN:-0}"

GPU_IMAGE="github-runner-gpu-2022-09-29-1664451806"
GPU_DISK_SIZE_GB=100
CPU_IMAGE="github-runner-cpu-2022-09-29-1664451255"
CPU_DISK_SIZE_GB=100

PROD_TEMPLATE_BASE_NAME="github-runner"
PROD_TEMPLATE_CONFIG_REPO="iree-org/iree"

TEMPLATE_CONFIG_REPO="${TEMPLATE_CONFIG_REPO:-${PROD_TEMPLATE_CONFIG_REPO}}"
TEMPLATE_CONFIG_REF="${TEMPLATE_CONFIG_REF:-$(git rev-parse HEAD)}"
TEMPLATE_BASE_NAME="${TEMPLATE_BASE_NAME:-${PROD_TEMPLATE_BASE_NAME}}"

if (( TESTING==0 )) && ! git merge-base --is-ancestor "${TEMPLATE_CONFIG_REF}" main; then
  echo "Creating testing template because TEMPLATE_CONFIG_REF='${TEMPLATE_CONFIG_REF}' is not on the main branch"
  TESTING=1
fi
if (( TESTING==0 )) && [[ "${TEMPLATE_CONFIG_REPO}" != "${PROD_TEMPLATE_CONFIG_REPO}" ]]; then
  echo "Creating testing template because TEMPLATE_CONFIG_REPO '${TEMPLATE_CONFIG_REPO}'!='${PROD_TEMPLATE_CONFIG_REPO}'"
  TESTING=1
fi
if (( TESTING==0 )) && [[ "${TEMPLATE_BASE_NAME}" != "${PROD_TEMPLATE_BASE_NAME}" ]]; then
  echo "Creating testing template because TEMPLATE_BASE_NAME '${TEMPLATE_BASE_NAME}'!='${PROD_TEMPLATE_BASE_NAME}'"
  TESTING=1
fi


# We need something to avoid duplicate names. Occasional collisions aren't that
# big a deal (user can just re-run), so we use few characters here, as the whole
# template name can only be 63 characters. We used to use the unix timestamp,
# but that's pretty long and not very human readable. We also used to include
# the date in ISO8601 format, but again, character limits and creation date is
# in the template metadata.
# 3 legal characters. Note this is using bash to expand the braced sequences.
SUFFIX="$(shuf --echo --head-count=3 --repeat {a..z} {0..9} | tr -d '\n')"
SHORT_REF="${TEMPLATE_CONFIG_REF:0:10}"
VERSION="${SHORT_REF}-${SUFFIX}"
if (( TESTING!=0 )); then
  VERSION="${VERSION}-testing"
fi
GITHUB_RUNNER_SCOPE=iree-org
GITHUB_RUNNER_VERSION="2.298.2"
GITHUB_RUNNER_ARCHIVE_DIGEST="0bfd792196ce0ec6f1c65d2a9ad00215b2926ef2c416b8d97615265194477117"
GITHUB_TOKEN_PROXY_URL="https://ght-proxy-zbhz5clunq-ue.a.run.app"

declare -a METADATA=(
  "github-runner-version=${GITHUB_RUNNER_VERSION}"
  "github-runner-archive-digest=${GITHUB_RUNNER_ARCHIVE_DIGEST}"
  "github-runner-config-ref=${TEMPLATE_CONFIG_REF}"
  "github-runner-config-repo=${TEMPLATE_CONFIG_REPO}"
  "github-runner-scope=${GITHUB_RUNNER_SCOPE}"
  "github-token-proxy-url=${GITHUB_TOKEN_PROXY_URL}"
)

declare -a common_args=(
  --project=iree-oss
  # `address=''` indicates an ephemeral IP. This *shouldn't* be necessary here,
  # as the gcloud docs say that this is the default, but in fact if you leave it
  # off the VM gets no external IP and is impossible to SSH into. This knowledge
  # was hard won.
  --network-interface=network=default,address='',network-tier=PREMIUM
  # Matches firewall rule for health check traffic
  --tags="allow-health-checks"
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
    "github-runner-type=${type}"
  )

  # Join on commas
  local metadata_string="$(IFS="," ; echo "${metadata[*]}")"

  local -a cmd=(
    gcloud compute instance-templates create
    "${TEMPLATE_BASE_NAME}-${group}-${type}-${VERSION}"
    "${common_args[@]}"
    --service-account="github-runner-${trust}-trust@iree-oss.iam.gserviceaccount.com"
    --metadata="${metadata_string}"
  )

  if [[ "${type}" == gpu ]]; then
    cmd+=(
      --machine-type=a2-highgpu-1g
      --maintenance-policy=TERMINATE
      --accelerator=count=1,type=nvidia-tesla-a100
      --create-disk="auto-delete=yes,boot=yes,image=projects/iree-oss/global/images/${GPU_IMAGE},mode=rw,size=${GPU_DISK_SIZE_GB},type=pd-balanced"
    )
  elif [[ "${type}" == cpu ]]; then
    cmd+=(
      --machine-type=n1-standard-96
      --maintenance-policy=MIGRATE
      --create-disk="auto-delete=yes,boot=yes,image=projects/iree-oss/global/images/${CPU_IMAGE},mode=rw,size=${CPU_DISK_SIZE_GB},type=pd-balanced"
    )
  else
    echo "Got unrecognized type '${type}'" >2
    exit 1
  fi
  if (( DRY_RUN==1 )); then
    # Prefix the command with a noop. It will still be printed by set -x
    cmd=(":" "${cmd[@]}")
  fi
  (set -x; "${cmd[@]}")
  echo ''
}

for group in presubmit postsubmit; do
  for type in gpu cpu; do
    create_template "${group}" "${type}"
  done
done
echo "Created new templates for version: ${VERSION}"

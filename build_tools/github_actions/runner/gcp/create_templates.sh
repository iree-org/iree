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
TESTING_SELF_DELETER="${TESTING_SELF_DELETER:-0}"

GPU_IMAGE="${GPU_IMAGE:-github-runner-gpu-2023-01-30-1675109292}"
GPU_DISK_SIZE_GB="${GPU_DISK_SIZE_GB:-100}"
CPU_IMAGE="${CPU_IMAGE:-github-runner-cpu-2023-01-30-1675109033}"
CPU_DISK_SIZE_GB="${CPU_DISK_SIZE_GB:-100}"

PROD_TEMPLATE_CONFIG_REPO="${PROD_TEMPLATE_CONFIG_REPO:-openxla/iree}"
GITHUB_RUNNER_SCOPE="${GITHUB_RUNNER_SCOPE:-openxla}"

TEMPLATE_CONFIG_REPO="${TEMPLATE_CONFIG_REPO:-${PROD_TEMPLATE_CONFIG_REPO}}"
TEMPLATE_CONFIG_REF="${TEMPLATE_CONFIG_REF:-$(git rev-parse HEAD)}"
TEMPLATE_NAME_PREFIX="${TEMPLATE_NAME_PREFIX:-gh-runner}"

if (( TESTING==0 )) && ! git merge-base --is-ancestor "${TEMPLATE_CONFIG_REF}" main; then
  echo "Creating testing template because TEMPLATE_CONFIG_REF='${TEMPLATE_CONFIG_REF}' is not on the main branch" >&2
  TESTING=1
fi
if (( TESTING==0 )) && [[ "${TEMPLATE_CONFIG_REPO}" != "${PROD_TEMPLATE_CONFIG_REPO}" ]]; then
  echo "Creating testing template because TEMPLATE_CONFIG_REPO '${TEMPLATE_CONFIG_REPO}'!='${PROD_TEMPLATE_CONFIG_REPO}'" >&2
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
GITHUB_RUNNER_VERSION="${GITHUB_RUNNER_VERSION:-2.303.0}"
GITHUB_RUNNER_ARCHIVE_DIGEST="${GITHUB_RUNNER_ARCHIVE_DIGEST:-e4a9fb7269c1a156eb5d5369232d0cd62e06bec2fd2b321600e85ac914a9cc73}"
GITHUB_TOKEN_PROXY_URL="${GITHUB_TOKEN_PROXY_URL:-https://ght-proxy-openxla-zbhz5clunq-ue.a.run.ap}"

if (( TESTING_SELF_DELETER==1 )); then
  INSTANCE_SELF_DELETER_URL="https://instance-self-deleter-testing-zbhz5clunq-uc.a.run.app"
else
  INSTANCE_SELF_DELETER_URL="https://instance-self-deleter-zbhz5clunq-uc.a.run.app"
fi

declare -a METADATA=(
  "github-runner-version=${GITHUB_RUNNER_VERSION}"
  "github-runner-archive-digest=${GITHUB_RUNNER_ARCHIVE_DIGEST}"
  "github-runner-config-ref=${TEMPLATE_CONFIG_REF}"
  "github-runner-config-repo=${TEMPLATE_CONFIG_REPO}"
  "github-runner-scope=${GITHUB_RUNNER_SCOPE}"
  "github-token-proxy-url=${GITHUB_TOKEN_PROXY_URL}"
  "instance-self-deleter-url=${INSTANCE_SELF_DELETER_URL}"
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
  # The instance group manager handles this for us and this is necessary to
  # achieve better local SSD performance:
  # https://cloud.google.com/compute/docs/disks/optimizing-local-ssd-performance#disable-automatic-restart
  --no-restart-on-failure
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
    --quiet
  )

  cmd+=(
    "${TEMPLATE_NAME_PREFIX}-${group}-${type}-${VERSION}"
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
      # See comment in build_tools/github_actions/runner/config/setup.sh
      --local-ssd=interface=NVME
    )
  elif [[ "${type}" == cpu ]]; then
    cmd+=(
      --machine-type=n1-standard-96
      --maintenance-policy=MIGRATE
      --create-disk="auto-delete=yes,boot=yes,image=projects/iree-oss/global/images/${CPU_IMAGE},mode=rw,size=${CPU_DISK_SIZE_GB},type=pd-balanced"
    )
  elif [[ "${type}" == c2s16 ]]; then
    cmd+=(
      --machine-type=c2-standard-16
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

  (set -x; "${cmd[@]}") >&2
  echo '' >&2
}

for group in presubmit postsubmit; do
  for type in gpu cpu c2s16; do
    create_template "${group}" "${type}"
  done
done

echo "Created new templates for version: ${VERSION}" >&2
echo "${VERSION}"

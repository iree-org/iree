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

GPU_IMAGE="${GPU_IMAGE:-github-runner-gpu-2023-10-13-1697221547}"
CPU_IMAGE="${CPU_IMAGE:-github-runner-cpu-2023-06-02-1685725199}"
ARM64_IMAGE="${ARM64_IMAGE:-github-runner-arm64-2023-11-01-1698857095}"
DISK_SIZE_GB="${DISK_SIZE_GB:-1000}"

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
GITHUB_RUNNER_VERSION="${GITHUB_RUNNER_VERSION:-2.311.0}"
GITHUB_RUNNER_X64_ARCHIVE_DIGEST="${GITHUB_RUNNER_X64_ARCHIVE_DIGEST:-29fc8cf2dab4c195bb147384e7e2c94cfd4d4022c793b346a6175435265aa278}"
GITHUB_RUNNER_ARM64_ARCHIVE_DIGEST="${GITHUB_RUNNER_ARM64_ARCHIVE_DIGEST:-5d13b77e0aa5306b6c03e234ad1da4d9c6aa7831d26fd7e37a3656e77153611e}"
GITHUB_TOKEN_PROXY_URL="${GITHUB_TOKEN_PROXY_URL:-https://ght-proxy-openxla-zbhz5clunq-ue.a.run.app}"

if (( TESTING_SELF_DELETER==1 )); then
  INSTANCE_SELF_DELETER_URL="https://instance-self-deleter-testing-zbhz5clunq-uc.a.run.app"
else
  INSTANCE_SELF_DELETER_URL="https://instance-self-deleter-zbhz5clunq-uc.a.run.app"
fi

declare -a METADATA=(
  "github-runner-version=${GITHUB_RUNNER_VERSION}"
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
  # The instance group manager handles this for us
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

  if [[ "${type}" == "arm64" ]]; then
    local runner_arch="arm64"
    local runner_archive_digest="${GITHUB_RUNNER_ARM64_ARCHIVE_DIGEST}"
  else
    local runner_arch="x64"
    local runner_archive_digest="${GITHUB_RUNNER_X64_ARCHIVE_DIGEST}"
  fi
  metadata+=(
    "github-runner-archive-url=https://github.com/actions/runner/releases/download/v${GITHUB_RUNNER_VERSION}/actions-runner-linux-${runner_arch}-${GITHUB_RUNNER_VERSION}.tar.gz"
    "github-runner-archive-digest=${runner_archive_digest}"
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

  case "${type}" in
    gpu)
      cmd+=(
        --machine-type=n1-standard-16
        --maintenance-policy=TERMINATE
        --accelerator=count=1,type=nvidia-tesla-t4
        --create-disk="auto-delete=yes,boot=yes,image=projects/iree-oss/global/images/${GPU_IMAGE},mode=rw,size=${DISK_SIZE_GB},type=pd-ssd"
      )
      ;;
    a100)
      cmd+=(
        --machine-type=a2-highgpu-1g
        --maintenance-policy=TERMINATE
        --accelerator=count=1,type=nvidia-tesla-a100
        --create-disk="auto-delete=yes,boot=yes,image=projects/iree-oss/global/images/${GPU_IMAGE},mode=rw,size=${DISK_SIZE_GB},type=pd-ssd"
      )
      ;;
    cpu)
      cmd+=(
        --machine-type=n1-standard-96
        --maintenance-policy=MIGRATE
        --create-disk="auto-delete=yes,boot=yes,image=projects/iree-oss/global/images/${CPU_IMAGE},mode=rw,size=${DISK_SIZE_GB},type=pd-ssd"
      )
      ;;
    c2s601t)
      cmd+=(
        --machine-type=c2-standard-60
        --threads-per-core=1
        --maintenance-policy=MIGRATE
        --create-disk="auto-delete=yes,boot=yes,image=projects/iree-oss/global/images/${CPU_IMAGE},mode=rw,size=${DISK_SIZE_GB},type=pd-ssd"
      )
      ;;
    arm64)
      cmd+=(
        --machine-type=t2a-standard-8
        --maintenance-policy=MIGRATE
        --create-disk="auto-delete=yes,boot=yes,image=projects/iree-oss/global/images/${ARM64_IMAGE},mode=rw,size=${DISK_SIZE_GB},type=pd-ssd"
      )
      ;;
    *)
      echo "Got unrecognized type '${type}'" >2
      exit 1
      ;;
  esac

  if (( DRY_RUN==1 )); then
    # Prefix the command with a noop. It will still be printed by set -x
    cmd=(":" "${cmd[@]}")
  fi

  (set -x; "${cmd[@]}") >&2
  echo '' >&2
}

for group in presubmit postsubmit; do
  # TODO(#14661): Remove c2s601t if we decide not to migrate benchmarks to it.
  for type in gpu a100 cpu c2s601t arm64; do
    create_template "${group}" "${type}"
  done
done

echo "Created new templates for version: ${VERSION}" >&2
echo "${VERSION}"

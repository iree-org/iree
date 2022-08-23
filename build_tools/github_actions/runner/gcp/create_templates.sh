#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";

TEMPLATE_BASE_NAME="${TEMPLATE_BASE_NAME:-github-runner}"
TIME_STRING="$(date +%Y-%m-%d-%s)"
REPO="${TEMPLATE_CONFIG_REPO:-iree-org/iree}"
REF="${TEMPLATE_CONFIG_REF:-$(git rev-parse HEAD)}"
SHORT_REF="${REF:0:10}"
STARTUP_SCRIPT_PATH="/tmp/startup_script.${SHORT_REF}.sh"

sed -E "s/CONFIG_REF=main/CONFIG_REF=${REF}/" -E "s@REPO=iree-org/iree@REPO=${REPO}@" "${SCRIPT_DIR}/startup_script.sh" > "${STARTUP_SCRIPT_PATH}"

declare -a common_args=(
  --project=iree-oss
  --network-interface=network=default,network-tier=PREMIUM
  --provisioning-model=STANDARD
  --scopes=https://www.googleapis.com/auth/cloud-platform
  --no-shielded-secure-boot
  --shielded-vtpm
  --shielded-integrity-monitoring
  --reservation-affinity=any
  --metadata-from-file=startup-script="${STARTUP_SCRIPT_PATH}"
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

  local -a args=(
    "${TEMPLATE_BASE_NAME}-${group}-${type}-${SHORT_REF}-${TIME_STRING}"
    "${common_args[@]}"
    --service-account="github-runner-${trust}-trust@iree-oss.iam.gserviceaccount.com"
    --metadata="github-runner-group=${group},github-runner-trust=${trust},github-runner-labels=${type},runner-config-ref=${REF},github-runner-scope=iree-org,github-token-proxy-url=https://ght-proxy-zbhz5clunq-ue.a.run.app"
  )

  local disk_name="${TEMPLATE_BASE_NAME}-${group}-${type}-${SHORT_REF}-${TIME_STRING}"

  if [[ "${type}" == gpu ]]; then
    args+=(
      --machine-type=a2-highgpu-1g
      --maintenance-policy=TERMINATE
      --accelerator=count=1,type=nvidia-tesla-a100
      --create-disk="auto-delete=yes,boot=yes,device-name=${disk_name},image=projects/iree-oss/global/images/github-runner-gpu-2022-08-15-1660603500,mode=rw,size=1000,type=pd-balanced"
    )
  elif [[ "${type}" == cpu ]]; then
    args+=(
      --machine-type=n1-standard-96
      --maintenance-policy=MIGRATE
      --create-disk="auto-delete=yes,boot=yes,device-name=${disk_name},image=projects/iree-oss/global/images/github-runner-2022-07-28-1659048799,mode=rw,size=1000,type=pd-balanced"
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

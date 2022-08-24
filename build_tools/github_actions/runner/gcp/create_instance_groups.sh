#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This is a WIP script for creating managed instance groups. It isn't really a
# proper script, since most everything is hardcoded. Mostly it's a way to
# document the necessary commands. In due course, it may be turned into a proper
# script, probably in Python, but for now just edit the environment variables at
# the top.

set -euo pipefail

# For now, just change these parameters
VERSION=5d33335129-2022-08-23-1661292386
REGION=us-west1
ZONES=us-west1-a,us-west1-b,us-west1-c
AUTOSCALING=0
MIN_SIZE=1
MAX_SIZE=10

function create_mig() {
  local runner_group="$1"
  local type="$2"

  GROUP_NAME="github-runner-${runner_group}-${type}-${REGION}"
  TEMPLATE="github-runner-${runner_group}-${type}-${VERSION}"

  local -a create_args=(
    "${GROUP_NAME}"
    --project=iree-oss
    --base-instance-name="${GROUP_NAME}"
    --size="${MIN_SIZE}"
    --template="${TEMPLATE}"
    --zones="${ZONES}"
    --target-distribution-shape=EVEN
  )

  (set -x; gcloud beta compute instance-groups managed create "${create_args[@]}")
  echo ""

  if (( AUTOSCALING == 1 )) && [[ "${type}" == cpu ]]; then
    local -a autoscaling_args=(
      "${GROUP_NAME}"
      --project=iree-oss
      --region="${REGION}"
      --cool-down-period=60
      --min-num-replicas="${MIN_SIZE}"
      --max-num-replicas="${MAX_SIZE}"
      --mode=only-scale-out
      --target-cpu-utilization=0.6
    )


    (set -x; gcloud beta compute instance-groups managed set-autoscaling "${autoscaling_args[@]}")
    echo ""
  fi
}

create_mig presubmit cpu

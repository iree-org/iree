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

VERSION="${VERSION:-7138511883-62g-testing}"
REGION="${REGION:-us-west1}"
ZONES="${ZONES:-us-west1-a,us-west1-b,us-west1-c}"
AUTOSCALING="${AUTOSCALING:-1}"
GROUP="${GROUP:-presubmit}"
TYPE="${TYPE:-cpu}"
MIG_NAME_PREFIX="${MIG_NAME_PREFIX:-gh-runner}"
TEMPLATE_NAME_PREFIX="${TEMPLATE_NAME_PREFIX:-gh-runner}"
DRY_RUN="${DRY_RUN:-0}"

# For GPU groups, these should both be set to the target group size, as
# autoscaling currently does not work for these.
MIN_SIZE="${MIN_SIZE:-3}"
MAX_SIZE="${MAX_SIZE:-3}"
# Whether this is a testing MIG (i.e. not prod)
TESTING="${TESTING:-1}"

if (( TESTING==0 )) && [[ "${VERSION}" == *testing* ]]; then
  echo "Creating testing mig because VERSION='${VERSION}' contains 'testing'" >&2
  TESTING=1
fi

function create_mig() {
  local runner_group="$1"
  local type="$2"

  local mig_name="${MIG_NAME_PREFIX}"
  if (( TESTING == 1 )); then
    mig_name+="-testing"
  fi
  mig_name+="-${runner_group}-${type}-${REGION}"
  template="${TEMPLATE_NAME_PREFIX}-${runner_group}-${type}-${VERSION}"

  local -a create_cmd=(
    gcloud beta compute instance-groups managed create
    "${mig_name}"
    --project=iree-oss
    --base-instance-name="${mig_name}"
    --size="${MIN_SIZE}"
    --template="${template}"
    --zones="${ZONES}"
    --region="${REGION}"
    --target-distribution-shape=EVEN
    --health-check=http-8080-ok
  )

  if (( DRY_RUN==1 )); then
    # Prefix the command with a noop. It will still be printed by set -x
    create_cmd=(":" "${create_cmd[@]}")
  fi

  (set -x; "${create_cmd[@]}") >&2
  echo '' >&2

  local -a autoscaling_cmd=(
    gcloud beta compute instance-groups managed set-autoscaling
    "${mig_name}"
    --project=iree-oss
    --region="${REGION}"
    --cool-down-period=60
    --min-num-replicas="${MIN_SIZE}"
    --max-num-replicas="${MAX_SIZE}"
    --mode=only-scale-out
    --target-cpu-utilization=0.6
  )

  if (( DRY_RUN==1 )); then
    # Prefix the command with a noop. It will still be printed by set -x
    autoscaling_cmd=(":" "${autoscaling_cmd[@]}")
  fi

  (set -x; "${autoscaling_cmd[@]}") >&2
  echo '' >&2
}

create_mig "${GROUP}" "${TYPE}"

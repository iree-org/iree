#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This is a WIP script for canarying updates to managed instance groups. It
# isn't really a proper script, since most everything is hardcoded. Mostly it's
# a way to document the necessary commands. In due course, it may be turned into
# a proper script, probably in Python, but for now just edit the environment
# variables at the top.

set -euo pipefail

OLD_VERSION=deadbeef12-2022-08-23-1661292386
NEW_VERSION=deadbeef34-2022-08-23-1661296461
# If this MIG is for testing (i.e. not prod)
TESTING=1

RUNNER_GROUP=presubmit
TYPE=cpu
REGION=us-west1

function canary_update() {
  local runner_group="$1"
  local type="$2"
  local region="$3"

  local mig_name="github-runner"
  if (( TESTING == 1 )); then
    mig_name+="-testing"
  fi
  mig_name+="-${runner_group}-${type}-${region}"

  gcloud compute instance-groups managed rolling-action start-update \
    "${mig_name}" \
    --version=template="github-runner-${runner_group}-${type}-${OLD_VERSION}",name=orig \
    --canary-version=template="github-runner-${runner_group}-${type}-${NEW_VERSION}",target-size=10%,name=canary \
    --type=opportunistic \
    --region="${REGION}"
}

canary_update "${RUNNER_GROUP}" "${TYPE}" "${REGION}"

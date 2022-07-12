#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";

source "${SCRIPT_DIR}/functions.sh"

ALLOWED_EVENTS=(
  "push"
  "workflow_dispatch"
  "release"
  "deployment"
  "deployment_status"
  "schedule"
)

if ! isContained "${GITHUB_EVENT_NAME}" "${ALLOWED_EVENTS[@]}"; then
  echo "Event type '${GITHUB_EVENT_NAME}' is not allowed on this runner. Aborting workflow."
  # clean up any nefarious stuff we may have fetched in job setup
  cd /home/runner/actions-runner/_work
  rm -rf _actions/ _temp/
  exit 1
fi

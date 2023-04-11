#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

source /runner-root/config/functions.sh

ALLOWED_EVENTS=(
  "push"
  "workflow_dispatch"
  "release"
  "deployment"
  "deployment_status"
  "schedule"
)

if ! is_contained "${GITHUB_EVENT_NAME}" "${ALLOWED_EVENTS[@]}"; then
  echo "Event type '${GITHUB_EVENT_NAME}' is not allowed on this runner. Aborting workflow."
  # clean up any nefarious stuff we may have fetched in job setup. This
  # shouldn't be necessary with ephemeral runners, but shouldn't hurt either.
  cd /runner-root/actions-runner/_work
  rm -rfv _actions/ _temp/
  exit 1
fi

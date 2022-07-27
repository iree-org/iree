#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Starts the GitHub actions runner as a custom service.

set -euo pipefail

SELF_NAME="$(basename $0)"
SELF_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"
SELF_PATH="${SELF_DIR}/${SELF_NAME}"

>&2 echo "Starting runner"

/dev/shm/actions-runner/bin/Runner.Listener run --startuptype service

CODE=$?

# A return code 3 means the runner stopped because it is updating. Wait a few
# seconds to let it update and then respawn.
if [[ $code == 3 ]]; then
  >&2 echo "Runner received update"
  sleep 10
  exec ${SELF_PATH}
else
  exit ${CODE}
fi

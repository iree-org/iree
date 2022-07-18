#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";

source "${SCRIPT_DIR}/functions.sh"

RUNNER_GROUP="$(get_attribute github-runner-group)"

"/home/runner/validate_trigger.${RUNNER_GROUP}.sh"
/home/runner/chown_workdir.sh

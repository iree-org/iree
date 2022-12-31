#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

source /runner-root/config/functions.sh

RUNNER_GROUP="$(get_attribute github-runner-group)"

"/runner-root/config/hooks/validate_trigger.${RUNNER_GROUP}.sh"
/runner-root/config/hooks/chown_workdir.sh

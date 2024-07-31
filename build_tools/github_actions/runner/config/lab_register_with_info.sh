#!/bin/bash

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -xeuo pipefail

# Uncomment the lines below and populate with proper values.
# export CPU_PLATFORM=""
# export GPU_PLATFORM=""
# export MACHINE_TYPE=""
# export RUNNER_NAME=""
# export RUNNER_TRUST="<minimal / basic>"
# export RUNNER_GROUP="<presubmit / postsubmit>"

./lab_register.sh

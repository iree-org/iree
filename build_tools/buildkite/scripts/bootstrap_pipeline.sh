#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Uploads the given pipeline if it hasn't already been bootstrapped. Common
# usage is for a pipeline to call this on itself.

set -euo pipefail

PIPELINE_FILE=${1?}

if [[ "${PIPELINE_BOOTSTRAPPED}" == "false" ]]; then
  export PIPELINE_BOOTSTRAPPED="true"
  buildkite-agent pipeline upload --replace $PIPELINE_FILE
fi

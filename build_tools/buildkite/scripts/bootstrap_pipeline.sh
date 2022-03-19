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

if [[ -z "${PIPELINE_BOOTSTRAPPED:+x}" ]]; then
  exit 0
fi

export PIPELINE_BOOTSTRAPPED=1

buildkite-agent pipeline upload --replace $PIPELINE_FILE

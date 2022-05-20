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

# Skip when running locally because Buildkite cli doesn't support replacing
# pipelines (https://github.com/buildkite/cli/issues/122). This isn't too big a
# limitation because presumably we're already running with the local file that
# would be bootstrapped.
if [[ ${BUILDKITE_ORGANIZATION_SLUG} == "local" ]]; then
  echo "Local run. Skipping bootstrapping."
  exit 0
fi

if [[ "${PIPELINE_BOOTSTRAPPED}" == "false" ]]; then
  export PIPELINE_BOOTSTRAPPED="true"
  buildkite-agent pipeline upload --replace $PIPELINE_FILE
fi

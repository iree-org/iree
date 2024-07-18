#!/bin/bash

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Pre-fetches docker images so we don't have to wait for docker to fetch them as
# part of `docker run` in the CI jobs. This isn't comprehensive but pre-fetches
# things we think are pretty likely to be useful to pre-fetch. It runs in the
# background so shouldn't have a significant impact on other startup tasks. It
# also won't block fetches being done by the job itself. It could compete for
# network bandwidth though, so we don't want to just always try to fetch
# everything.

set -euo pipefail

source /runner-root/config/functions.sh

nice_curl https://raw.githubusercontent.com/iree-org/iree/main/build_tools/docker/prod_digests.txt \
  --output /tmp/prod_digests.txt

# Basically everything uses a derivative of one of these
grep 'gcr.io/iree-oss/base@' /tmp/prod_digests.txt | xargs docker pull
grep 'gcr.io/iree-oss/base-bleeding-edge@' /tmp/prod_digests.txt | xargs docker pull

RUNNER_TYPE="$(get_attribute github-runner-type)"

if [[ "${RUNNER_TYPE}" == gpu ]]; then
  grep 'gcr.io/iree-oss/nvidia@' /tmp/prod_digests.txt | xargs docker pull
fi

#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build and test IREE's core within the gcr.io/iree-oss/bazel image using
# Kokoro.
# Requires the environment variables KOKORO_ROOT and KOKORO_ARTIFACTS_DIR, which
# are set by Kokoro.

set -x
set -e
set -o pipefail

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

# This doesn't really need everything in the frontends image, but we want the
# cache to be shared with the integrations build (no point building LLVM twice)
# and the cache key is the docker container it's run in (to ensure correct cache
# hits).
"${KOKORO_ARTIFACTS_DIR?}/github/iree/build_tools/kokoro/gcp_ubuntu/docker_run.sh" \
  gcr.io/iree-oss/frontends-swiftshader@sha256:3090418a8d8a64c356d35eff285af32570a72f41127aa123209c1562f57abb01 \
  build_tools/kokoro/gcp_ubuntu/bazel/linux/x86-swiftshader/core/build.sh

# Kokoro will rsync this entire directory back to the executor orchestrating the
# build which takes forever and is totally useless.
rm -rf "${KOKORO_ARTIFACTS_DIR?}"/*

# Print out artifacts dir contents after deleting them as a coherence check.
ls -1a "${KOKORO_ARTIFACTS_DIR?}/"

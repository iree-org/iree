#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile the project towards Android arm64-v8a with the
# gcr.io/iree-oss/android image using Kokoro.
# Requires the environment variables KOKORO_ROOT and KOKORO_ARTIFACTS_DIR, which
# are set by Kokoro.

set -x
set -e
set -o pipefail

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

source "${KOKORO_ARTIFACTS_DIR?}/github/iree/build_tools/kokoro/gcp_ubuntu/docker_common.sh"

# Sets DOCKER_RUN_ARGS
docker_setup

# Need to use frontends image (which also has Android toolchain) to build the
# TFLite compiler for generating benchmarks.
docker run "${DOCKER_RUN_ARGS[@]?}" \
  gcr.io/iree-oss/frontends@sha256:080d9f833bc772b53add2f4ad4ba06723cfb56a05f3a47e12e08c3de95308576 \
  build_tools/kokoro/gcp_ubuntu/cmake/android/build.sh arm64-v8a

# Kokoro will rsync this entire directory back to the executor orchestrating the
# build which takes forever and is totally useless.
rm -rf "${KOKORO_ARTIFACTS_DIR?}"/*

# Print out artifacts dir contents after deleting them as a coherence check.
ls -1a "${KOKORO_ARTIFACTS_DIR?}/"

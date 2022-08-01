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

set -xeuo pipefail

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

# Need to use frontends image (which also has Android toolchain) to build the
# TFLite compiler for generating benchmarks.
"${KOKORO_ARTIFACTS_DIR?}/github/iree/build_tools/kokoro/gcp_ubuntu/docker_run.sh" \
  gcr.io/iree-oss/frontends@sha256:bad174c580cdefaf435ce31a7df6bdd7f7cb7bfdcdff5d1acf40f630acf85bf5 \
  build_tools/kokoro/gcp_ubuntu/cmake/android/build.sh arm64-v8a

# Kokoro will rsync this entire directory back to the executor orchestrating the
# build which takes forever and is totally useless.
rm -rf "${KOKORO_ARTIFACTS_DIR?}"/*

# Print out artifacts dir contents after deleting them as a coherence check.
ls -1a "${KOKORO_ARTIFACTS_DIR?}/"

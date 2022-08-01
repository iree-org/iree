#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build and test the project within the gcr.io/iree-oss/swiftshader
# image using Kokoro.
# Requires the environment variables KOKORO_ROOT and KOKORO_ARTIFACTS_DIR, which
# are set by Kokoro.

set -x
set -e
set -o pipefail

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

"${KOKORO_ARTIFACTS_DIR?}/github/iree/build_tools/kokoro/gcp_ubuntu/docker_run.sh" \
  gcr.io/iree-oss/swiftshader@sha256:5027d56cdfee743d956bffd035668f7784166a486c48c74b42e5882cb0c289bf \
  build_tools/kokoro/gcp_ubuntu/cmake/linux/x86-swiftshader/build.sh

# Kokoro will rsync this entire directory back to the executor orchestrating the
# build which takes forever and is totally useless.
rm -rf "${KOKORO_ARTIFACTS_DIR?}"/*

# Print out artifacts dir contents after deleting them as a coherence check.
ls -1a "${KOKORO_ARTIFACTS_DIR?}/"

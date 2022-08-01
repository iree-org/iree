#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build and test the python bindings and frontend integrations on GPU within
# gcr.io/iree-oss/frontends-nvidia
# Requires the environment variables KOKORO_ROOT and KOKORO_ARTIFACTS_DIR, which
# are set by Kokoro.

set -x
set -e
set -o pipefail

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

# Print NVIDIA GPU information inside the VM
dmesg | grep NVRM
dpkg -l | grep nvidia
nvidia-smi || true

"${KOKORO_ARTIFACTS_DIR?}/github/iree/build_tools/kokoro/gcp_ubuntu/docker_run.sh" \
  --gpus all \
  gcr.io/iree-oss/frontends-nvidia@sha256:e934ed09e9e60c28ebe11a02f37a993dd975db40118d410c4279d0fa2d4e6b9a \
  build_tools/kokoro/gcp_ubuntu/cmake-bazel/linux/x86-turing/build.sh

# Kokoro will rsync this entire directory back to the executor orchestrating the
# build which takes forever and is totally useless.
rm -rf "${KOKORO_ARTIFACTS_DIR?}"/*

# Print out artifacts dir contents after deleting them as a coherence check.
ls -1a "${KOKORO_ARTIFACTS_DIR?}/"

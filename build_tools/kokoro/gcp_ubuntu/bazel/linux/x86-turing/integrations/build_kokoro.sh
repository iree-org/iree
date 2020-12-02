#!/bin/bash

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build and test IREE's integrations within the gcr.io/iree-oss/bazel-nvidia
# image using Kokoro.

set -e
set -x

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

# Kokoro checks out the repository here.
WORKDIR="${KOKORO_ARTIFACTS_DIR?}/github/iree"

source "${KOKORO_ARTIFACTS_DIR?}/github/iree/build_tools/kokoro/gcp_ubuntu/docker_common.sh"

docker_setup

# The image hash is pinned to an older version than that reflected by the Docker
# hierarchy because updating nvidia/vulkan causes everything to fail for some
# reason.
# TODO(#3550): Allow this to follow the checked-in Docker hierarchy.
docker run "${DOCKER_RUN_ARGS[@]?}" \
  --gpus all \
  gcr.io/iree-oss/bazel-tensorflow-nvidia@sha256:575ba235ebbbcee5bc26f20c6362664a62113ac869c8868ad415c175fe9c08b0 \
  build_tools/kokoro/gcp_ubuntu/bazel/linux/x86-turing/integrations/build.sh

# Kokoro will rsync this entire directory back to the executor orchestrating the
# build which takes forever and is totally useless.
sudo rm -rf "${KOKORO_ARTIFACTS_DIR?}"/*

# Print out artifacts dir contents after deleting them as a coherence check.
ls -1a "${KOKORO_ARTIFACTS_DIR?}/"

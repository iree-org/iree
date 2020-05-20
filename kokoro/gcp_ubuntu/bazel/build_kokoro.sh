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

# Build and test the project within the gcr.io/iree-oss/bazel-tensorflow
# image using Kokoro.

set -e
set -x

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

# Kokoro checks out the repository here.
WORKDIR="${KOKORO_ARTIFACTS_DIR?}/github/iree"

# Mount the checked out repository, make that the working directory and run the
# tests in the bazel-tensorflow image.
docker run \
  --volume "${WORKDIR?}:${WORKDIR?}" \
  --workdir="${WORKDIR?}" \
  --rm \
  gcr.io/iree-oss/bazel-tensorflow@sha256:18e4d648ebd367cfab7596fff769639d526cffc1f526be88488660ff4575f3ce \
  kokoro/gcp_ubuntu/bazel/build.sh

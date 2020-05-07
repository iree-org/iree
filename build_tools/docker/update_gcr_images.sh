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

# Builds and pushes bazel and bazel-tensorflow images to gcr.io/iree-oss/ from
# within the GCP IREE-OSS project.

set -x
set -e

# Ensure correct authorization.
gcloud auth configure-docker

# Build and push the bazel image.
docker build --tag bazel build_tools/docker/bazel/
docker tag bazel gcr.io/iree-oss/bazel
docker push gcr.io/iree-oss/bazel

# Build and push the bazel-tensorflow image.
docker build --tag bazel-tensorflow build_tools/docker/bazel_tensorflow/
docker tag bazel-tensorflow gcr.io/iree-oss/bazel-tensorflow
docker push gcr.io/iree-oss/bazel-tensorflow

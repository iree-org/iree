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

# Builds and pushes the rbe-toolchain image to gcr.io/iree-oss/

set -x
set -e

# Ensure correct authorization.
gcloud auth configure-docker

# Build and push the rbe-toolchain image.
docker build --tag gcr.io/iree-oss/rbe-toolchain build_tools/docker/rbe_toolchain/
docker push gcr.io/iree-oss/rbe-toolchain

echo '
Remember to update the rbe_default digest in the WORKSPACE file to reflect the
new ID for the container.

Use `docker images --digests` to view the ID.'

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

# An image for building IREE with bazel and running tests on a Kokoro VM.

# Build using:
# docker build \
#  --tag gcr.io/iree-oss/bazel-kokoro \
#  --file build_tools/docker/bazel_kokoro/Dockerfile \
#  .

# Set up the image and working directory.
FROM gcr.io/iree-oss/bazel-tensorflow
WORKDIR /usr/src/git/iree/

# Copy the PR that Kokoro checked out into the image.
COPY . .
ENTRYPOINT ["kokoro/gcp_ubuntu/bazel/build.sh"]

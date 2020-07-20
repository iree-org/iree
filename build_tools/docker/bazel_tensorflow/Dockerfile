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

# An image for building IREE with tensorflow integrations using bazel.

# Build using:
# docker build --tag gcr.io/iree-oss/bazel-tensorflow \
#   build_tools/docker/bazel_tensorflow/

# Run interactively using the following, where IREE_WORKDIR is the path to your
# local dev environment:
# docker run -it --rm --entrypoint bash --volume ${IREE_WORKDIR}:/usr/src/iree/ \
#   gcr.io/iree-oss/bazel-tensorflow

# Set up the image and working directory.
FROM gcr.io/iree-oss/bazel-bindings

# Install tensorflow.
RUN python3 -m pip install tf-nightly

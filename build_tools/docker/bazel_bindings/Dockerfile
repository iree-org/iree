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

# An image for building IREE with python bindings using bazel.

# Build using:
# docker build --tag gcr.io/iree-oss/bazel-bindings \
#   build_tools/docker/bazel_bindings/

# Run interactively using the following, where IREE_WORKDIR is the path to your
# local dev environment:
# docker run -it --rm --entrypoint bash --volume ${IREE_WORKDIR}:/usr/src/iree/ \
#   gcr.io/iree-oss/bazel-bindings

# Set up the image and working directory.
FROM gcr.io/iree-oss/bazel

# Install python3 and numpy.
RUN apt-get update \
  && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
  && python3 -m pip install --upgrade pip \
  && python3 -m pip install numpy

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

# An image for building IREE using bazel.

# Build using:
# docker build --tag gcr.io/iree-oss/bazel build_tools/docker/bazel/

# Run interactively using the following, where IREE_WORKDIR is the path to your
# local dev environment:
# docker run -it --rm --entrypoint bash \
#   --volume "${IREE_WORKDIR?}:/usr/src/iree/" \
#   gcr.io/iree-oss/bazel

# Set up the image and working directory.
FROM ubuntu:18.04
WORKDIR /usr/src/iree

RUN apt-get update

# Set environment variables.
ENV CXX clang++
ENV CC clang
ENV PYTHON_BIN /usr/bin/python3
ENV IREE_LLVMAOT_LINKER_PATH /usr/bin/ld

# Install git for updating IREE's submodules.
RUN apt-get install -y git

# Install Bazel.
# https://docs.bazel.build/versions/master/install-ubuntu.html
ARG BAZEL_VERSION=2.1.0
RUN apt-get install -y unzip zip wget \
  && wget "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION?}/bazel-${BAZEL_VERSION?}-installer-linux-x86_64.sh" \
  && chmod +x "bazel-${BAZEL_VERSION?}-installer-linux-x86_64.sh" \
  && "./bazel-${BAZEL_VERSION?}-installer-linux-x86_64.sh" --user \
  && rm "bazel-${BAZEL_VERSION?}-installer-linux-x86_64.sh"
# Install a newer version of Bazel. We don't need the full installation now.
# Just need to provide a different version for the version-identification
# wrapper script to find in /root/.bazel/bin
ARG NEW_BAZEL_VERSION=3.3.1
RUN cd "/root/.bazel/bin" \
  && wget "https://releases.bazel.build/${NEW_BAZEL_VERSION?}/release/bazel-${NEW_BAZEL_VERSION?}-linux-x86_64" \
  && chmod +x "bazel-${NEW_BAZEL_VERSION?}-linux-x86_64"
# ENV does not allow ${variable?} syntax.
ENV PATH "/root/bin:${PATH}"

# Install core IREE dependencies.
RUN apt-get install -y clang libsdl2-dev

# Install python2 numpy. Temporary fix for issue #1737:
# https://github.com/google/iree/issues/1737
RUN apt-get install -y python-pip \
  && python -m pip install --upgrade pip \
  && python -m pip install numpy

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

# Set environment variables.
ENV CXX clang++
ENV CC clang
ENV PYTHON_BIN /usr/bin/python3
ENV IREE_LLVMAOT_LINKER_PATH /usr/bin/ld

RUN apt-get update \
  && apt-get install -y \
    # git for updating IREE's submodules.
    git \
    # utilities for later installations
    unzip \
    zip \
    wget \
    # core IREE dependencies.
    clang \
    libsdl2-dev

# Disable apt-key parse waring. If someone knows how to do whatever the "proper"
# thing is then feel free. The warning complains about parsing apt-key output,
# which we're not even doing.
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

# Install Bazel.
# https://docs.bazel.build/versions/master/install-ubuntu.html
ARG BAZEL_VERSION=3.3.1
# Change to a new version if upgrading Bazel.
ARG NEW_BAZEL_VERSION=3.3.1
RUN wget -qO - https://bazel.build/bazel-release.pub.gpg | apt-key add - \
  && echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" \
    | tee /etc/apt/sources.list.d/bazel.list \
  && apt-get update \
  # Install Bazel pinned at the version we want. Optionally install an
  # additional version of Bazel to ease upgrades (modify NEW_BAZEL_VERSION
  # above). Bazel does some shenanigans to select the correct version based on
  # your .bazelversion file. When upgrading, we therefore need to have both the
  # old and new version. When the versions are the same this second installation
  # is effectively a noop.
  && apt-get install "bazel=${BAZEL_VERSION?}" "bazel-${NEW_BAZEL_VERSION?}"

# TF requires python2 numpy at configure time...
# TODO(#1737): Remove this
RUN apt-get update \
    && apt-get install -y python-pip \
    && python -m pip install --upgrade pip \
    && python -m pip install numpy

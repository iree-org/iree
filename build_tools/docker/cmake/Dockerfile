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

# An image for building IREE using CMake.

# Build using:
# docker build --tag gcr.io/iree-oss/cmake build_tools/docker/cmake/

# Run interactively using the following, where IREE_WORKDIR is the path to your
# local dev environment:
# docker run -it --rm --entrypoint bash \
#   --volume "${IREE_WORKDIR?}:/usr/src/iree/" \
#   gcr.io/iree-oss/cmake

# Set up the image and working directory.
FROM ubuntu:18.04
WORKDIR /usr/src/iree/

RUN apt-get update
# TODO: Remove this if the `apt-get install` below works without it again.
RUN apt update

# Update cmake to v3.13+, which is ahead of apt-get's version (3.10.2).
# Install dependencies, including an old version of cmake to bootstrap.
ENV CMAKE_VERSION 3.13.5
RUN apt-get install -y clang cmake libssl-dev wget \
  && mkdir ./cmake_install \
  && cd cmake_install \
  && wget "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION?}/cmake-${CMAKE_VERSION?}.tar.gz" \
  && tar -xzvf "cmake-${CMAKE_VERSION?}.tar.gz" \
  && cd "cmake-${CMAKE_VERSION?}/" \
  && cmake . \
  && make \
  && make install

# Install dependencies.
RUN apt-get install -y \
  git \
  ninja-build \
  python3 \
  python3-pip \
  python3-setuptools \
  # Install dependencies for the python bindings tests.
  && python3 -m pip install --upgrade pip \
  && python3 -m pip install numpy absl-py

# Environment variables for IREE.
ENV CC /usr/bin/clang
ENV CXX /usr/bin/clang++
ENV IREE_LLVMAOT_LINKER_PATH /usr/bin/ld

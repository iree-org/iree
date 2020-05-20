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

# Run using:
# docker run -it --rm --entrypoint bash \
#   --volume /home/phoenix/iree/fork:/usr/src/iree/ \
#   gcr.io/iree-oss/cmake

# Build and test IREE.
# cmake -G Ninja -B build/ \
#   -DCMAKE_C_COMPILER=clang \
#   -DCMAKE_CXX_COMPILER=clang++ \
#   -DIREE_BUILD_PYTHON_BINDINGS=ON .
# cmake --build build/
# ./build_tools/cmake/test.sh


# Set up the image and working directory.
FROM ubuntu:18.04
WORKDIR /usr/src/iree/

RUN apt-get update

# Update cmake to v3.13+, which is ahead of apt-get's version (3.10.2).
# Install dependencies, including an old version of cmake to bootstrap.
RUN apt-get install -y clang cmake libssl-dev wget
RUN mkdir ./cmake_install \
  && cd cmake_install \
  && wget https://github.com/Kitware/CMake/releases/download/v3.17.2/cmake-3.17.2.tar.gz \
  && tar -xzvf cmake-3.17.2.tar.gz \
  && cd cmake-3.17.2/ \
  && cmake . \
  && make \
  && make install

# Install dependencies.
RUN apt-get install -y git
RUN apt-get install -y ninja-build
RUN apt-get install -y python3 python3-pip python3-setuptools

# Install dependencies for the python bindings tests.
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install numpy absl-py

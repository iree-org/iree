#!/bin/bash

# Copyright 2019 Google LLC
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

# Build the project with bazel using Kokoro.

set -e

set -x

# TODO(b/145697435) Kokoro VMs have bad public keys. Delete when this is fixed.
echo "Deleting corrupted ppa sources"
sudo rm -rf /etc/apt/sources.list.d/nvidia-docker.list*
# Make sure we don't run the old version of bazel the VM comes with.
echo "Deleting old bazel version"
sudo rm /usr/local/bin/bazel

export BAZEL_VERSION=1.1.0
echo "Installing bazel ${BAZEL_VERSION}"
# https://docs.bazel.build/versions/master/install-ubuntu.html
wget "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh"
chmod +x "bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh"
./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh --user
export PATH="${HOME}/bin:${PATH}"
bazel --version

echo "Installing clang 6.0"
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main"
sudo apt-get update
sudo apt-get install -y clang-6.0
clang++-6.0 --version

python3 -V

echo "Preparing environment variables"
export CXX=clang++-6.0
export CC=clang-6.0
export PYTHON_BIN="$(which python3)"

# Kokoro checks out the repository here.
cd ${KOKORO_ARTIFACTS_DIR}/github/iree
echo "Checking out submodules"
git submodule update --init --depth 1000 --jobs 8

echo "Building and testing with bazel"
./build_tools/bazel_build.sh

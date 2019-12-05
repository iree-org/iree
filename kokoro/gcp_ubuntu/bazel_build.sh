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

# Having separate build scripts with this indirection is recommended by the
# Kokoro setup instructions.

set -e

set -x

# Hackery to allow SSHing into the VM for debugging 
# echo "ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBEHXMM+6DzYMf3u/EEGGYKANayUoPLNFR7uQO/qYaw/jI+Nj+sYmN/fB8CAQTG/Dvjpzl8kMXLMqbDFuSALA4OA= you@gnubby.key" >> ~/.ssh/authorized_keys
# external_ip=$(curl -s -H "Metadata-Flavor: Google" http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip)
# echo "INSTANCE_EXTERNAL_IP=${external_ip}"
# sleep 3600 # Give me an hour to fiddle with this

echo "Deleting corrupted ppa sources"
sudo rm -rf /etc/apt/sources.list.d/nvidia-docker.list*

echo "Installing bazel $BAZEL_VERSION"
export BAZEL_VERSION=1.1.0
# https://docs.bazel.build/versions/master/install-ubuntu.html
wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
chmod +x bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh --user
export PATH=$HOME/bin:$PATH
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

echo "$CXX"
echo "$CC"
echo "$PYTHON_BIN"

# Kokoro checks out the repository here.
cd ${KOKORO_ARTIFACTS_DIR}/github/iree
echo "Checking out submodules"
git submodule update --init --depth 1000 --jobs 8
./build_tools/bazel_build.sh

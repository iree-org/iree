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

echo "Installing bazel $BAZEL_VERSION"
export BAZEL_VERSION=1.1.0
# https://docs.bazel.build/versions/master/install-ubuntu.html
sudo apt-get install unzip zip
wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
chmod +x bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh --user
rm bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
export PATH=$HOME/bin:$PATH
bazel --version

echo "Installing clang"
sudo apt-get install clang || true # Continue even if this fails
clang --version
clang++ --version

#echo "Installing python"
#sudo apt-get install python3 python3-pip
python3 -V
#sudo pip3 install numpy

export CXX=clang++
export CC=clang
export PYTHON_BIN="$(which python3)"


echo "$CXX"
echo "$CC"
echo "$PYTHON_BIN"

# Kokoro checks out the repository here.
cd ${KOKORO_ARTIFACTS_DIR}/github/iree
./build_tools/bazel_build.sh

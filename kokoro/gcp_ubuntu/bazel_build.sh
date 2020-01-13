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

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

# Check these exist and print the versions for later debugging
bazel --version
python3 -V

export PYTHON_BIN="$(which python3)"

# Kokoro checks out the repository here.
cd ${KOKORO_ARTIFACTS_DIR?}/github/iree
echo "Checking out submodules"
git submodule update --init --depth 1000 --jobs 8

echo "Building and testing with bazel"
./build_tools/scripts/bazel_build.sh

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

# Build the IREE documentation with CMake. Designed for CI, but can be run
# manually. This uses previously cached build results and does not clear the
# build directory.

set -x
set -e

ROOT_DIR=$(git rev-parse --show-toplevel)
BUILD_DIR=${BUILD_DIR:-"build-docs"}

CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}

"$CMAKE_BIN" --version
ninja --version

cd ${ROOT_DIR?}
if [ -d ${BUILD_DIR} ]
then
  echo "Build directory already exists. Will use cached results there."
else
  echo "Build directory does not already exist. Creating a new one."
  mkdir ${BUILD_DIR}
fi
cd ${BUILD_DIR}

"$CMAKE_BIN" .. -DCMAKE_BUILD_TYPE=Release \
                -DIREE_BUILD_COMPILER=ON \
                -DIREE_BUILD_TESTS=ON \
                -DIREE_BUILD_SAMPLES=OFF \
                -DIREE_BUILD_DOCS=ON \
                -DIREE_BUILD_DEBUGGER=OFF \
                -DIREE_BUILD_PYTHON_BINDINGS=OFF \
                -G Ninja
ninja iree-doc

cd ${ROOT_DIR?}

# Update op_coverage.md
scripts/update_op_coverage.py ${BUILD_DIR}

# Update e2e_coverage.md
PYTHON_BIN=`which python3` scripts/update_e2e_coverage.py ${BUILD_DIR}

# Copy a curated list of docs to publish. This is expected to cover all docs
# under docs/ after they are refreshed.

cp README.md ${BUILD_DIR}/doc/index.md
cp docs/IREE-Architecture.svg ${BUILD_DIR}/doc/

cp docs/roadmap.md ${BUILD_DIR}/doc/
cp docs/roadmap_design.md ${BUILD_DIR}/doc/
cp docs/developer_overview.md ${BUILD_DIR}/doc/
cp docs/testing_guide.md ${BUILD_DIR}/doc/
cp docs/iree_community.md ${BUILD_DIR}/doc/

mkdir -p ${BUILD_DIR}/doc/GetStarted/
cp docs/GetStarted/getting_started_windows_bazel.md ${BUILD_DIR}/doc/GetStarted/
cp docs/GetStarted/getting_started_windows_cmake.md ${BUILD_DIR}/doc/GetStarted/
cp docs/GetStarted/getting_started_windows_vulkan.md ${BUILD_DIR}/doc/GetStarted/
cp docs/GetStarted/getting_started_linux_bazel.md ${BUILD_DIR}/doc/GetStarted/
cp docs/GetStarted/getting_started_linux_cmake.md ${BUILD_DIR}/doc/GetStarted/
cp docs/GetStarted/getting_started_linux_vulkan.md ${BUILD_DIR}/doc/GetStarted/
cp docs/GetStarted/getting_started_macos_bazel.md ${BUILD_DIR}/doc/GetStarted/
cp docs/GetStarted/getting_started_macos_cmake.md ${BUILD_DIR}/doc/GetStarted/
cp docs/GetStarted/getting_started_android_cmake.md ${BUILD_DIR}/doc/GetStarted/
cp docs/GetStarted/getting_started_python.md ${BUILD_DIR}/doc/GetStarted/
cp docs/GetStarted/generic_vulkan_env_setup.md ${BUILD_DIR}/doc/GetStarted/
cp docs/GetStarted/cmake_options_and_variables.md ${BUILD_DIR}/doc/GetStarted/

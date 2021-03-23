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

# Build the IREE project with CMake. Designed for CI, but can be run manually.
# This uses previously cached build results and does not clear the build
# directory. For a clean build, use build_tools/cmake/clean_build.sh

set -x
set -e

ROOT_DIR=$(git rev-parse --show-toplevel)

CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}

"$CMAKE_BIN" --version
ninja --version

cd ${ROOT_DIR?}
if [ -d "build" ]
then
  echo "Build directory already exists. Will use cached results there."
else
  echo "Build directory does not already exist. Creating a new one."
  mkdir build
fi
cd build

CMAKE_ARGS=(
  "-G" "Ninja"
  "-DCMAKE_BUILD_TYPE=RelWithDebInfo"

  # Enable docs build on the CI. The additional builds are pretty fast and
  # give us early warnings for some types of website publication errors.
  "-DIREE_BUILD_DOCS=ON"

  # Enable building the python bindings on CI. Most heavy targets are gated on
  # IREE_ENABLE_TENSORFLOW, so what's left here should be fast.
  "-DIREE_BUILD_PYTHON_BINDINGS=ON"
)

"$CMAKE_BIN" "${CMAKE_ARGS[@]?}" ..
"$CMAKE_BIN" --build .

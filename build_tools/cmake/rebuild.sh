# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
  # Let's make linking fast
  "-DIREE_ENABLE_LLD=ON"
  "-DCMAKE_BUILD_TYPE=RelWithDebInfo"

  # Enable docs build on the CI. The additional builds are pretty fast and
  # give us early warnings for some types of website publication errors.
  "-DIREE_BUILD_DOCS=ON"

  # Enable building the python bindings on CI.
  "-DIREE_BUILD_PYTHON_BINDINGS=ON"

  # Enable assertions. We wouldn't want to be testing *only* with assertions
  # enabled, but at the moment only certain CI builds are using this script,
  # e.g. ASan builds are not using this, so by enabling assertions here, we
  # get a reasonable mix of {builds with asserts, builds with other features
  # such as ASan but without asserts}.
  "-DIREE_ENABLE_ASSERTIONS=ON"
)

"$CMAKE_BIN" "${CMAKE_ARGS[@]?}" "$@" ..
echo "Building all"
echo "------------"
"$CMAKE_BIN" --build . -- -k 0

echo "Building test deps"
echo "------------------"
"$CMAKE_BIN" --build . --target iree-test-deps -- -k 0

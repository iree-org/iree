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

  # Enable building the python bindings on CI. Most heavy targets are gated on
  # IREE_ENABLE_TENSORFLOW, so what's left here should be fast.
  "-DIREE_BUILD_PYTHON_BINDINGS=ON"
)

"$CMAKE_BIN" "${CMAKE_ARGS[@]?}" ..
"$CMAKE_BIN" --build .

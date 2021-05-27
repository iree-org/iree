#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile the project towards Android with CMake using Kokoro.

set -e
set -x

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <android-abi>"
  exit 1
fi

ANDROID_ABI="${1?}"

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

# Check these exist and print the versions for later debugging
export CMAKE_BIN="$(which cmake)"
"${CMAKE_BIN?}" --version
"${CC?}" --version
"${CXX?}" --version
python3 --version
echo "Android NDK path: $ANDROID_NDK"

echo "Initializing submodules"
./scripts/git/submodule_versions.py init

echo "Cross-compiling with cmake"
./build_tools/cmake/build_android.sh $ANDROID_ABI

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

# Populates a bare git reference repository for IREE and deps, suitable
# for use in the clone --reference= argument.
#
# CAUTION: This is experimental and it is very easy to hurt yourself. If A
# reference repo is corrupted/moved/deleted/etc, then all dependent
# repositories will be rendered unusable.
#
# Initial setup (very slow):
#   mkdir /path/to/refrepo
#   cd /path/to/refrepo
#   git init --bare
#   ./path/to/populate_reference_repo.sh
#
# Subsequently, the populate_reference_repo.sh command can be run again to
# fetch updates.
#
# Then a git clone or submodule update can be done, consulting such a repo:
#   git clone --reference=/path/to/refrepo https://github.com/google/iree.git
#   cd iree
#   git submodule init
#   git submodule update --reference=/path/to/refrepo
set -ex

# Create a temporary directory.
TOPTEMPDIR=`mktemp -d`
trap "rm -Rf $TOPTEMPDIR" EXIT

function populate_repo() {
  local repo="$1"
  local url="$2"

  if git remote get-url "$repo"; then
    echo "Fetching updates from $repo ($url)..."
    git fetch "$repo"
  else
    # Do a two step clone to avoid all refs being sent to the server.
    echo "Cloning fresh from $repo ($url) into temp directory..."
    # First clone the directory separately.
    TEMPDIR="$TOPTEMPDIR/$repo"
    mkdir "$TEMPDIR"
    git clone "$url" "$TEMPDIR"

    # Then fetch from the temporary dir into our main repo.
    echo "Fetching into reference repo..."
    git remote add "$repo" "$TEMPDIR/"
    git fetch "$repo"

    # Then change the remote URL and fetch normally.
    echo "Cleaning up..."
    git remote set-url "$repo" "$url"
    git fetch "$repo"
    rm -Rf "$TEMPDIR"
  fi
}

populate_repo iree https://github.com/google/iree.git
populate_repo tf https://github.com/tensorflow/tensorflow.git
populate_repo gtest https://github.com/google/googletest.git
populate_repo absl https://github.com/abseil/abseil-cpp.git
populate_repo llvm https://github.com/llvm/llvm-project.git
populate_repo mlir https://github.com/tensorflow/mlir.git
populate_repo flatbuffers https://github.com/google/flatbuffers.git
populate_repo vk https://github.com/KhronosGroup/Vulkan-Headers.git
populate_repo vkmem https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
populate_repo gemmlowp https://github.com/google/gemmlowp.git
populate_repo google_tracing_framework https://github.com/google/tracing-framework.git
populate_repo glslang https://github.com/KhronosGroup/glslang.git
populate_repo spirv_tools https://github.com/KhronosGroup/SPIRV-Tools.git
populate_repo spirv_headers https://github.com/KhronosGroup/SPIRV-Headers.git


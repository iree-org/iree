#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# While IREE can be built with a normal CUDA Toolkit installation, the compiler
# and runtime only actually depend on a very thin (platform independent) slice
# of it. Since a full install is very high overhead, this script downloads
# official NVIDIA distribution zip artifacts and extracts what we need to
# a location that our build is setup to handle. This is intended for CI
# systems but can also be used by end users who would like a foolproof
# way to make IREE minimally buildable with CUDA support.
#
# See: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#tarball-zipfile-overview
#
# Usage:
#
# For personal use, fetch to your home directory (the IREE CMake build is
# configured to probe for an installation here):
#   ./fetch_cuda_deps.sh $HOME/.iree_cuda_deps
#
# For system-wide (i.e. docker images):
#   ./fetch_cuda_deps.sh /usr/local/iree_cuda_deps
set -e

ARCH="$(uname -m)"
if [[ "${ARCH}" == "aarch64" ]]; then
  echo "ERROR: Script does not support ${ARCH}."
  exit 1
fi

TARGET_DIR="$1"
if [ -z "$TARGET_DIR" ]; then
  echo "ERROR: Expected target directory (typically /usr/local/iree_cuda_deps for CI or $HOME/.iree_cuda_deps for local)"
  exit 1
fi

DOWNLOAD_SCRIPT_URL="https://raw.githubusercontent.com/NVIDIA/build-system-archive-import-examples/main/parse_redist.py"
DOWNLOAD_DIR="$TARGET_DIR/download"
DOWNLOAD_SCRIPT_PATH="$DOWNLOAD_DIR/parse_redist.py"

# Parameters to the download script.
# Look for an appropriate redistrib_*.json here to verify:
#   https://developer.download.nvidia.com/compute/cuda/redist/
VERSION="12.2.1"
PRODUCT="cuda"
OS="linux"
ARCH="x86_64"

# Components that we need to fetch.
COMPONENTS=(
  cuda_cccl   # CXX Core Compute Libraries
  cuda_nvcc   # CUDA NVCC
  cuda_cudart # CUDA Runtime
)

# Paths within the arch specific installation that we want to retain.
RETAIN_PATHS=(
  LICENSE
  nvvm/libdevice/libdevice.10.bc
  include
)

echo "Extracting to $TARGET_DIR"
mkdir -p "$TARGET_DIR"
mkdir -p "$DOWNLOAD_DIR"

# First fetch the download script to the tmp dir.
cd "$DOWNLOAD_DIR"
echo "Fetching download script from $DOWNLOAD_SCRIPT_URL"
curl -L "$DOWNLOAD_SCRIPT_URL" > "$DOWNLOAD_SCRIPT_PATH"

# Then use the download script to fetch and flatten each component we want
# into the tmp dir.
# This will produce a unified directory tree under:
#   flat/$OS-$ARCH
SRC_DIR="$DOWNLOAD_DIR/flat/${OS}-${ARCH}"
for component in ${COMPONENTS[@]}; do
  echo "Downloading component $component"
  python3 "$DOWNLOAD_SCRIPT_PATH" \
    --label "$VERSION" \
    --product "$PRODUCT" \
    --os "$OS" \
    --arch "$ARCH" \
    --component "$component"

  # Each component comes with a LICENSE file that we need to be able to
  # overwrite.
  chmod +w "${SRC_DIR}/LICENSE"
done

if ! [ -d "$SRC_DIR" ]; then
  echo "ERROR: Download did not produce expected source dir: $SRC_DIR"
  exit 1
fi

for rel_path in ${RETAIN_PATHS[@]}; do
  src_file="$SRC_DIR/$rel_path"
  target_file="$TARGET_DIR/$rel_path"
  echo "Copy $src_file -> $target_file"
  mkdir -p "$(dirname $target_file)"
  cp -Rf $src_file $target_file
done

# Delete tmp directory (saves ~100MiB in docker images).
rm -Rf "$DOWNLOAD_DIR"

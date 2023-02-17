#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

LLVM_VERSION="$1"
LINUX_VERSION="$(lsb_release -ds)"

declare -a PACKAGES=(
    "clang-${LLVM_VERSION}"
    "lld-${LLVM_VERSION}"
    # IREE transitive dependencies
    # Next time someone's updating these, try dropping sdl2. It shouldn't be
    # necessary anymore, but you need to make sure that we get `vulkaninfo` from
    # another source (this is currently installing it implicitly).
    libsdl2-dev
    libssl-dev
    # A much better CMake builder
    ninja-build
    # Needed for building lld with Bazel (as currently configured)
    libxml2-dev
    # Optional for tools like llvm-symbolizer, which we could build from
    # source but would rather just have available ahead of time
    llvm-dev
)

# The current lowest supported version Ubuntu 18.04 can't build Tracy. Drop this
# check when we drop the support of Ubuntu 18.04.
if [[ "${LINUX_VERSION}" = "Ubuntu 18.04*" ]]; then
  PACKAGES+=(
    # Tracy build and run dependencies
    pkg-config
    libcapstone-dev
    libtbb-dev
    libzstd-dev
  )
fi

apt-get update
apt-get install -y "${PACKAGES[@]}"

# Being called exactly "lld" appears to be load bearing. Someone is welcome to
# tell me a better way to install a specific version as just lld (lld=<version>
# doesn't work).
ln -s "lld-${LLVM_VERSION}" /usr/bin/lld
ln -s "ld.lld-${LLVM_VERSION}" /usr/bin/ld.lld

#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

CCACHE_VERSION=4.7.4
ARCH="$(uname -m)"
if [[ "${ARCH}" == "x86_64" ]]; then
  curl --silent --show-error --fail --location \
      "https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-x86_64.tar.xz" \
      --output ccache.tar.xz \
      "https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-x86_64.tar.xz.asc" \
      --output ccache.tar.xz.asc \
      "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x5a939a71a46792cf57866a51996dda075594adb8" \
      --output JOEL_PUBKEY.asc
  gpg --import JOEL_PUBKEY.asc
  gpg --verify ccache.tar.xz.asc
  tar -xvf ccache.tar.xz --strip-components=1
  cp ccache /usr/bin/
elif [[ "${ARCH}" == "aarch64" ]]; then
  # Latest version of ccache is not released for arm64, built it
  git clone --depth 1 --branch "v${CCACHE_VERSION}" https://github.com/ccache/ccache.git
  mkdir -p ccache/build && cd "$_"
  cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ..
  ninja
  cp ccache /usr/bin/
fi


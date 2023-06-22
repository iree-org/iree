#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

CCACHE_VERSION=4.7.4
machine=$(uname -m)
if [[ $machine == x86_64 ]]; then
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
elif  [[ $machine == aarch64 ]]; then
  # Apt install for Ubuntu 20.04 gives ccache version 3.7.7, I assume you want
  # something newer. Two alternatives from what I can find
  #   1) archlinuxarm.org has a 4.8.2 version
  #   2) build from source
  # but I may be missing something...
  apt-get install -y ccache
fi


#!/bin/bash
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euox pipefail

BAZELISK_VERSION="$1"

ARCH="$(dpkg --print-architecture)"
BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/v${BAZELISK_VERSION}/bazelisk-linux-${ARCH}"

# Releases since v1.29.0 publish a SHA-256 next to each binary.
curl --silent --fail --show-error --location "${BAZELISK_URL}" --output bazelisk
curl --silent --fail --show-error --location "${BAZELISK_URL}.sha256" --output bazelisk.sha256
echo "$(cut -d ' ' -f 1 bazelisk.sha256)  bazelisk" | sha256sum -c -
rm ./bazelisk.sha256

cp ./bazelisk /usr/local/bin/bazel
chmod +x /usr/local/bin/bazel
cp ./bazelisk /usr/local/bin/bazelisk
chmod +x /usr/local/bin/bazelisk
rm ./bazelisk

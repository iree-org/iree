#!/bin/bash
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euox pipefail

BAZELISK_VERSION="$1"

ARCH="$(dpkg --print-architecture)"

curl --silent --fail --show-error --location \
    "https://github.com/bazelbuild/bazelisk/releases/download/v${BAZELISK_VERSION}/bazelisk-linux-${ARCH}" \
    --output bazelisk

cp ./bazelisk /usr/local/bin/bazel
chmod +x /usr/local/bin/bazel
cp ./bazelisk /usr/local/bin/bazelisk
chmod +x /usr/local/bin/bazelisk
rm ./bazelisk

#!/bin/bash
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euox pipefail

BAZELISK_VERSION="$1"

ARCH="$(dpkg --print-architecture)"

# SHA-256 of each supported bazelisk release binary.
case "${BAZELISK_VERSION}-${ARCH}" in
  1.21.0-amd64) BAZELISK_SHA256="655a5c675dacf3b7ef4970688b6a54598aa30cbaa0b9e717cd1412c1ef9ec5a7" ;;
  1.21.0-arm64) BAZELISK_SHA256="ff793b461968e30d9f954c080f4acaa557edbdeab1ce276c02e4929b767ead66" ;;
  *)
    echo "No SHA-256 known for bazelisk ${BAZELISK_VERSION} on ${ARCH}; add it to $0."
    exit 1
    ;;
esac

curl --silent --fail --show-error --location \
    "https://github.com/bazelbuild/bazelisk/releases/download/v${BAZELISK_VERSION}/bazelisk-linux-${ARCH}" \
    --output bazelisk
echo "${BAZELISK_SHA256}  bazelisk" | sha256sum -c -

cp ./bazelisk /usr/local/bin/bazel
chmod +x /usr/local/bin/bazel
cp ./bazelisk /usr/local/bin/bazelisk
chmod +x /usr/local/bin/bazelisk
rm ./bazelisk

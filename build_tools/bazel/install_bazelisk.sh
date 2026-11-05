#!/bin/bash
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euox pipefail

BAZELISK_VERSION="$1"

ARCH="$(dpkg --print-architecture)"

# SHA-256 of each supported binary, as published next to it in the release.
case "${BAZELISK_VERSION}-${ARCH}" in
  1.30.0-amd64) BAZELISK_SHA256="a248ae2b2b62b5baebb748f6fba7203bff1a27e5d01b55399a8406744cb539e8" ;;
  1.30.0-arm64) BAZELISK_SHA256="b2bcc040c213b44586b121bda6d983c35df121b917c3a7a66fb16f2fa0df1a4f" ;;
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

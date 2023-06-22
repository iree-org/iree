#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

if ! [[ -f .bazelversion ]]; then
  echo "Couldn't find .bazelversion file in current directory" >&2
  exit 1
fi

BAZEL_VERSION="$(cat .bazelversion)"

# We could do the whole apt install dance, but this technique works across a
# range of platforms, allowing us to use a single script. See
# https://bazel.build/install/ubuntu#binary-installer
machine=$(uname -m)
if [ "$machine" == "aarch64" ]; then
  machine="arm64"
fi

curl --silent --fail --show-error --location \
  "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION?}/bazel-${BAZEL_VERSION?}-linux-${machine}" \
  --output /usr/bin/bazel
chmod +x /usr/bin/bazel

if [[ "$(bazel --version)" != "bazel ${BAZEL_VERSION}" ]]; then
  echo "Bazel installation failed" >&2
  exit 1
fi

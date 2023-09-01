#!/bin/bash
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

ROCM_VERSION=$1
AMDGPU_VERSION=$2

ARCH="$(uname -m)"
if [[ "${ARCH}" == "x86_64" ]]; then
  apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl libnuma-dev gnupg \
    && curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
    && printf "deb [arch=amd64] https://repo.radeon.com/rocm/apt/${ROCM_VERSION}/ ubuntu main" | tee /etc/apt/sources.list.d/rocm.list \
    && printf "deb [arch=amd64] https://repo.radeon.com/amdgpu/${AMDGPU_VERSION}/ubuntu focal main" | tee /etc/apt/sources.list.d/amdgpu.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libelf1 \
    kmod \
    file \
    rocm-dev \
    build-essential
else
  echo "Installing ROCM for ${ARCH} is not supported yet."
fi

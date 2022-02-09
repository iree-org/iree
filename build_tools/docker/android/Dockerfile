# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for cross-compiling IREE towards Android.

FROM gcr.io/iree-oss/base@sha256:1e551dda4cfdc8f92dc13c4c2c8e5f5b79d3103a4534bcc002ef5cc02ee9dd4f
ARG NDK_VERSION=r21d
WORKDIR /install-ndk

ENV ANDROID_NDK "/usr/src/android-ndk-${NDK_VERSION}"

RUN wget -q "https://dl.google.com/android/repository/android-ndk-${NDK_VERSION?}-linux-x86_64.zip" \
    && unzip -q "android-ndk-${NDK_VERSION?}-linux-x86_64.zip" -d /usr/src/  \
    && rm -rf /install-ndk

WORKDIR /

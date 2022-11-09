# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for cross-compiling IREE towards Android.

FROM gcr.io/iree-oss/base@sha256:7c3027c48b94fc38e64488987fc7893c100526c57308d25cef0c6b76a2dfe117
ARG NDK_VERSION=r25b
WORKDIR /install-ndk

ENV ANDROID_NDK "/usr/src/android-ndk-${NDK_VERSION}"

RUN wget -q "https://dl.google.com/android/repository/android-ndk-${NDK_VERSION?}-linux.zip" \
    && unzip -q "android-ndk-${NDK_VERSION?}-linux.zip" -d /usr/src/  \
    && rm -rf /install-ndk

WORKDIR /

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for cross-compiling IREE towards Android using CMake.

ARG NDK_VERSION=r21d

FROM gcr.io/iree-oss/util@sha256:40846b4aea5886af3250399d6adfdb3e1195a8b0177706bb0375e812d62dc49c AS install-ndk
ARG NDK_VERSION
WORKDIR /install-ndk

RUN wget "https://dl.google.com/android/repository/android-ndk-${NDK_VERSION?}-linux-x86_64.zip"

RUN unzip "android-ndk-${NDK_VERSION?}-linux-x86_64.zip" -d /usr/src/

FROM gcr.io/iree-oss/cmake-python@sha256:51817f1a98f9ed9237577133b4c674b163280fd747c1745d6d0d93f0f2b01fb3 AS final
ARG NDK_VERSION
COPY --from=install-ndk "/usr/src/android-ndk-${NDK_VERSION}" "/usr/src/android-ndk-${NDK_VERSION}"
ENV ANDROID_NDK "/usr/src/android-ndk-${NDK_VERSION}"

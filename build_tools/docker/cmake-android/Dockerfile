# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for cross-compiling IREE towards Android using CMake.

ARG NDK_VERSION=r21d

FROM gcr.io/iree-oss/base@sha256:b8d9863c6ac913f167c6fab319d7cd883ab099312488709ee30b29976d63eb22 AS install-ndk
ARG NDK_VERSION
WORKDIR /install-ndk

RUN wget "https://dl.google.com/android/repository/android-ndk-${NDK_VERSION?}-linux-x86_64.zip"

RUN unzip -q "android-ndk-${NDK_VERSION?}-linux-x86_64.zip" -d /usr/src/

FROM gcr.io/iree-oss/base@sha256:b8d9863c6ac913f167c6fab319d7cd883ab099312488709ee30b29976d63eb22 AS final
ARG NDK_VERSION
COPY --from=install-ndk "/usr/src/android-ndk-${NDK_VERSION}" "/usr/src/android-ndk-${NDK_VERSION}"
ENV ANDROID_NDK "/usr/src/android-ndk-${NDK_VERSION}"

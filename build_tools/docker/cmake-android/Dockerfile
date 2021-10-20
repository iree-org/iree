# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for cross-compiling IREE towards Android using CMake.

ARG NDK_VERSION=r21d

FROM gcr.io/iree-oss/util@sha256:c9691ba3cd4287cee61be42103888d7c6f02c44c7c7b7d30a3dca7fc73ee92cb AS install-ndk
ARG NDK_VERSION
WORKDIR /install-ndk

RUN wget "https://dl.google.com/android/repository/android-ndk-${NDK_VERSION?}-linux-x86_64.zip"

RUN unzip "android-ndk-${NDK_VERSION?}-linux-x86_64.zip" -d /usr/src/

FROM gcr.io/iree-oss/cmake-python@sha256:7b805ffc94cdb93d3af2781566df42c55a16a60eb3fb3e97ced78712357f4813 AS final
ARG NDK_VERSION
COPY --from=install-ndk "/usr/src/android-ndk-${NDK_VERSION}" "/usr/src/android-ndk-${NDK_VERSION}"
ENV ANDROID_NDK "/usr/src/android-ndk-${NDK_VERSION}"

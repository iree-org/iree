# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/cmake-bazel-frontends@sha256:f8a911a4c6f8f2733f8fe9264992dcff192b01f4f5883a5cde1452fccc2fce5e AS final

# Note that NDK_VERSION should be kept in sync with the one in cmake-android.
# That actually means this isn't really an ARG because we cannot build this
# image with a different NDK version.
ARG NDK_VERSION=r21d
COPY --from=gcr.io/iree-oss/cmake-android@sha256:eb72a4864f30ba53868cf23caf44e60aca35a36682ed1b1a5fa37cd9fda8855c "/usr/src/android-ndk-${NDK_VERSION}" "/usr/src/android-ndk-${NDK_VERSION}"
ENV ANDROID_NDK "/usr/src/android-ndk-${NDK_VERSION}"


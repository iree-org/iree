# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/cmake-bazel-frontends@sha256:d4dd5cea9679bc3faaa25821160dc52afd51f32c3c5f7b960450cd66de49dd21 AS final

# Note that NDK_VERSION should be kept in sync with the one in cmake-android.
# That actually means this isn't really an ARG because we cannot build this
# image with a different NDK version.
ARG NDK_VERSION=r21d
COPY --from=gcr.io/iree-oss/cmake-android@sha256:7d780787608474301e74e1b5cc2a1bfd1304a79ed9e0774c7ed422c0e4a38625 "/usr/src/android-ndk-${NDK_VERSION}" "/usr/src/android-ndk-${NDK_VERSION}"
ENV ANDROID_NDK "/usr/src/android-ndk-${NDK_VERSION}"


# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/cmake-bazel-frontends@sha256:3c2945cf40699ac38d68ebd56b1ca9d72ed3e2fbe14f032a2a4fdc000b96b657 AS final

# Note that NDK_VERSION should be kept in sync with the one in cmake-android.
# That actually means this isn't really an ARG because we cannot build this
# image with a different NDK version.
ARG NDK_VERSION=r21d
COPY --from=gcr.io/iree-oss/cmake-android@sha256:15d3266ae4865f7642a4ef4d76e5181f0dc3482a7cfba9021b6b55be524208ec "/usr/src/android-ndk-${NDK_VERSION}" "/usr/src/android-ndk-${NDK_VERSION}"
ENV ANDROID_NDK "/usr/src/android-ndk-${NDK_VERSION}"


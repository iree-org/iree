# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This image provides the Vulkan SDK. Requires a child image to provide
# a Vulkan ICD.

FROM gcr.io/iree-oss/cmake-bazel-frontends@sha256:fe83ae0dd3f1948ee04e256ed0cfa9983b1cffe548cd36d53e5def490cca752e AS final

ARG VULKAN_SDK_VERSION=1.2.154.0

COPY --from=gcr.io/iree-oss/vulkan@sha256:5812ee64806a7f3df0739ccf0930c27cabce346901488eceb1ee66c9c0a5ae96 \
  /opt/vulkan-sdk/ /opt/vulkan-sdk/

ENV VULKAN_SDK="/opt/vulkan-sdk/${VULKAN_SDK_VERSION}/x86_64"

ENV PATH="${VULKAN_SDK}/bin:$PATH"

# Symlink the Vulkan loader to a system library directory. This is needed to
# allow Vulkan applications to find the Vulkan loader. It also avoids using
# LD_LIBRARY_PATH, which is not supported well by Docker.
RUN ln -s "${VULKAN_SDK}/lib/libvulkan.so" /usr/lib/x86_64-linux-gnu/ \
  && ln -s "${VULKAN_SDK}/lib/libvulkan.so.1" /usr/lib/x86_64-linux-gnu/

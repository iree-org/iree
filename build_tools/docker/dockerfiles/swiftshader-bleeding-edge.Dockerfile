# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/base-bleeding-edge@sha256:273559bafe04270a8ad5b21d31c6c056d2d966b5d06c41353da5063cc8a16616 AS install-swiftshader
WORKDIR /install-swiftshader

COPY build_tools/third_party/swiftshader/build_vk_swiftshader.sh ./
RUN ./build_vk_swiftshader.sh "/swiftshader" && rm -rf /install-swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json

WORKDIR /

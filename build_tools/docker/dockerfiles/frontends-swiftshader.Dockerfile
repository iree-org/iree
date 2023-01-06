# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/frontends@sha256:41e82bb62a3850c01d0252aa67aebd7816e47da0132af677cd70fc76731c4b5d
COPY --from=gcr.io/iree-oss/swiftshader@sha256:126f96acd2071d364ff09d1e726a543ded8689619dfd80ca63ce716b31a5c1d4 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json

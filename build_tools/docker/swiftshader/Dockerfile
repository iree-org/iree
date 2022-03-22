# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/base@sha256:d42da77c616301dacb109c901350b866401fa727cd0c3ddaf8f69418ecd22097 AS install-swiftshader
WORKDIR /install-swiftshader

RUN apt-get update && apt-get install -y git

ARG SWIFTSHADER_COMMIT=57eca9df5dc56d290f662ace7481ed758618d559

# zlib and xcb/shm.h are needed for compiling SwiftShader.
RUN apt-get update && apt-get install -y zlib1g-dev libxcb-shm0-dev

RUN git clone https://github.com/google/swiftshader
RUN cd swiftshader && git checkout "${SWIFTSHADER_COMMIT?}" && cd ..
# Only build SwiftShader Vulkan ICD.
RUN cmake -S swiftshader/ -B build-swiftshader/ \
           -GNinja \
           -DSWIFTSHADER_BUILD_TESTS=OFF
RUN cmake --build build-swiftshader/ \
           --config Release \
           --target vk_swiftshader
# Copy the ICD JSON and .so to a known place.
RUN cp -rf build-swiftshader/Linux /swiftshader
# Keep track of the commit we are using.
RUN echo "${SWIFTSHADER_COMMIT?}" > /swiftshader/git-commit

# Ubuntu 18.04
FROM gcr.io/iree-oss/base@sha256:d42da77c616301dacb109c901350b866401fa727cd0c3ddaf8f69418ecd22097 AS final
COPY --from=install-swiftshader /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json

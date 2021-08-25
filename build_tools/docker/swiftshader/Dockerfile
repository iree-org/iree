# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/cmake@sha256:9d9953acf5ca0cf1ff3e8de32f10f24dfab1c4e8ec5d1fc047f556024ee4bed6 AS install-swiftshader
WORKDIR /install-swiftshader

RUN apt-get update && apt-get install -y git

ARG SWIFTSHADER_COMMIT=e32890c54793664e21891a504793cc8c0c7a34fb

# zlib is needed for compiling SwiftShader.
RUN apt-get update && apt-get install -y zlib1g-dev
RUN git clone https://github.com/google/swiftshader
RUN cd swiftshader && git checkout "${SWIFTSHADER_COMMIT?}" && cd ..
# Only build SwiftShader Vulkan ICD.
RUN cmake -S swiftshader/ -B build-swiftshader/ \
           -GNinja \
           -DSWIFTSHADER_BUILD_VULKAN=ON \
           -DSWIFTSHADER_BUILD_EGL=OFF \
           -DSWIFTSHADER_BUILD_GLESv2=OFF \
           -DSWIFTSHADER_BUILD_GLES_CM=OFF \
           -DSWIFTSHADER_BUILD_PVR=OFF \
           -DSWIFTSHADER_BUILD_TESTS=OFF
RUN cmake --build build-swiftshader/ \
           --config Release \
           --target vk_swiftshader
# Copy the ICD JSON and .so to a known place.
RUN cp -rf build-swiftshader/Linux /swiftshader
# Keep track of the commit we are using.
RUN echo "${SWIFTSHADER_COMMIT?}" > /swiftshader/git-commit

# Ubuntu 18.04
FROM ubuntu@sha256:fd25e706f3dea2a5ff705dbc3353cf37f08307798f3e360a13e9385840f73fb3 AS final
COPY --from=install-swiftshader /swiftshader /swiftshader

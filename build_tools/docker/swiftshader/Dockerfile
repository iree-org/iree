# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM gcr.io/iree-oss/cmake AS install-swiftshader
WORKDIR /install-swiftshader

RUN apt-get update && apt-get install -y git

ARG SWIFTSHADER_COMMIT=6287c18b1d249152563f0cb2d5cb0c6d0eb9e3d6

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

FROM ubuntu:18.04 AS final
COPY --from=install-swiftshader /swiftshader /swiftshader

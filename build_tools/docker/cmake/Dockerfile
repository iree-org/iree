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

# An image for building IREE using CMake.

# These are separate args because there's no way to strip the patch version off
# to get the /usr/share path.
# See https://github.com/moby/moby/issues/41383
ARG CMAKE_MAJOR_VERSION=3
ARG CMAKE_MINOR_VERSION=13
ARG CMAKE_PATCH_VERSION=5

FROM gcr.io/iree-oss/util@sha256:40846b4aea5886af3250399d6adfdb3e1195a8b0177706bb0375e812d62dc49c AS install-cmake
ARG CMAKE_MAJOR_VERSION
ARG CMAKE_MINOR_VERSION
ARG CMAKE_PATCH_VERSION
ENV CMAKE_VERSION="${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}"
WORKDIR /install-cmake

# Install CMake v3.13, which is ahead of apt-get's version (3.10.2).
RUN wget "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION?}/cmake-${CMAKE_VERSION?}-Linux-x86_64.sh"
RUN chmod +x "./cmake-${CMAKE_VERSION?}-Linux-x86_64.sh"
RUN "./cmake-${CMAKE_VERSION?}-Linux-x86_64.sh" --skip-license --prefix=/usr/

FROM gcr.io/iree-oss/base@sha256:9b73f4e2b1239f65a19f2022e54f4b15310b805570831fbe2cf8b4dc928f1d10 AS final
ARG CMAKE_MAJOR_VERSION
ARG CMAKE_MINOR_VERSION

COPY --from=install-cmake /usr/bin/cmake /usr/bin/ctest /usr/bin/
COPY --from=install-cmake \
  "/usr/share/cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" \
  "/usr/share/cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}"

RUN apt-get update && apt-get install -y ninja-build

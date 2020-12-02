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

# An image for cross-compiling IREE towards Android using CMake.

ARG NDK_VERSION=r21d

FROM gcr.io/iree-oss/util@sha256:40846b4aea5886af3250399d6adfdb3e1195a8b0177706bb0375e812d62dc49c AS install-ndk
ARG NDK_VERSION
WORKDIR /install-ndk

RUN wget "https://dl.google.com/android/repository/android-ndk-${NDK_VERSION?}-linux-x86_64.zip"

RUN unzip "android-ndk-${NDK_VERSION?}-linux-x86_64.zip" -d /usr/src/

FROM gcr.io/iree-oss/cmake@sha256:644cc10ea5a33bd97be51a8f6fd6ee7e2ab3904f468873be0f71373b0ec48919 AS final
ARG NDK_VERSION
COPY --from=install-ndk "/usr/src/android-ndk-${NDK_VERSION}" "/usr/src/android-ndk-${NDK_VERSION}"
ENV ANDROID_NDK "/usr/src/android-ndk-${NDK_VERSION}"

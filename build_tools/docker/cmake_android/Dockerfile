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

# Build using:
# docker build --tag gcr.io/iree-oss/cmake-android \
#              build_tools/docker/cmake_android/

# Run interactively using the following, where IREE_WORKDIR is the path to your
# local dev environment:
# docker run -it --rm --entrypoint bash \
#   --volume "${IREE_WORKDIR?}:/usr/src/iree/" \
#   gcr.io/iree-oss/cmake-android

# Set up the image and working directory by inheriting the base
# CMake configuration.
FROM gcr.io/iree-oss/cmake

# Additionally, we need the Android NDK for cross compiling IREE towards
# Android. Download Android NDK r21d and set up environment variable.

ADD https://dl.google.com/android/repository/android-ndk-r21d-linux-x86_64.zip /usr/src/

RUN apt-get install -y unzip \
  && unzip /usr/src/android-ndk-r21d-linux-x86_64.zip -d /usr/src/ \
  && rm -f /usr/src/android-ndk-r21d-linux-x86_64.zip

ENV ANDROID_NDK /usr/src/android-ndk-r21d

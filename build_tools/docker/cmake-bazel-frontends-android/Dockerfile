# Copyright 2021 Google LLC
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

FROM gcr.io/iree-oss/cmake-bazel-frontends@sha256:699f6470e618c151f36cb4769ccaf2cfbf334ddb2fc2b6e69eb60732fe69bb6b AS final

# Note that NDK_VERSION should be kept in sync with the one in cmake-android.
# That actually means this isn't really an ARG because we cannot build this
# image with a different NDK version.
ARG NDK_VERSION=r21d
COPY --from=gcr.io/iree-oss/cmake-android@sha256:15d3266ae4865f7642a4ef4d76e5181f0dc3482a7cfba9021b6b55be524208ec "/usr/src/android-ndk-${NDK_VERSION}" "/usr/src/android-ndk-${NDK_VERSION}"
ENV ANDROID_NDK "/usr/src/android-ndk-${NDK_VERSION}"


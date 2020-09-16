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

# An image for building IREE and its Python bindings using CMake.

ARG CMAKE_MAJOR_VERSION=3
ARG CMAKE_MINOR_VERSION=13
ARG CMAKE_PATCH_VERSION=5

FROM gcr.io/iree-oss/cmake AS final
# Dependencies for the python bindings tests.
RUN apt-get update \
  && apt-get install -y \
    python3 \
    python3-pip \
    python3-setuptools \
  && python3 -m pip install --upgrade pip \
  && python3 -m pip install numpy absl-py

ENV PYTHON_BIN /usr/bin/python3

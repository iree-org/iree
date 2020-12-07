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

FROM ubuntu@sha256:fd25e706f3dea2a5ff705dbc3353cf37f08307798f3e360a13e9385840f73fb3 AS final

# Environment variables for IREE.
ENV CC /usr/bin/clang
ENV CXX /usr/bin/clang++
ENV IREE_LLVMAOT_LINKER_PATH /usr/bin/ld

RUN apt-get update \
  && apt-get install -y \
    # For updating IREE's submodules.
    git \
    # Core IREE dependencies.
    clang \
    libsdl2-dev \
    libssl-dev

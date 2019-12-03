#!/bin/bash

# Copyright 2019 Google LLC
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

# Build the project with bazel using Kokoro.

# Having separate build scripts with this indirection is recommended by the
# Kokoro setup instructions.

set -e

set -x

# Relative to the root of the repository. Kokoro will make sure we're in the
# ${KOKORO_ARTIFACTS_DIR}/github/iree directory (which is equivalent) before
# invoking this.

./build_tools/bazel_build.sh

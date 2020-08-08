#!/bin/bash

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

# Simulates the behavior of Kokoro on a local machine.
# Usage:
# ./build_tools/kokoro/gcp_ubuntu/simulate_kokoro.sh \
#   build_tools/kokoro/gcp_ubuntu/bazel/linux/x86/core/build.sh
#
# Just does the part of the Kokoro setup that we care about and invokes the
# given build script.

set -x
set -e
set -o pipefail

RELATIVE_KOKORO_BUILD_SCRIPT="${1?}"

# Set up the temporary Kokoro directories
kokoro_base="/dev/shm"
kokoro_dir="$(mktemp --directory --tmpdir kokoro-root-XXXXXX)"
kokoro_dir="$(echo $kokoro_dir | rev | cut -d/ -f1 | rev)"
export KOKORO_ROOT="${kokoro_base}/${kokoro_dir}"
mkdir -p "${KOKORO_ROOT?}/src/github"
export KOKORO_ARTIFACTS_DIR="${KOKORO_ROOT?}/src"

# "Clone" the repo
ln -s $(pwd) "${KOKORO_ARTIFACTS_DIR}/github/iree"
cd "${KOKORO_ARTIFACTS_DIR?}/github"

# The build script is assumed to be relative to the iree repo root.
KOKORO_BUILD_SCRIPT="${KOKORO_ARTIFACTS_DIR?}/github/iree/${RELATIVE_KOKORO_BUILD_SCRIPT?}"
chmod +x "${KOKORO_BUILD_SCRIPT?}"

# This is where Kokoro starts its execution.
cd "${KOKORO_ARTIFACTS_DIR?}"

# Run the actual script.
"${KOKORO_BUILD_SCRIPT?}"

# Clean up after ourselves.
rm -rf "${KOKORO_ROOT?}"

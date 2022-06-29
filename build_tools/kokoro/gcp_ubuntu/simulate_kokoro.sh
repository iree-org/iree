#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Simulates the behavior of Kokoro on a local machine.
# Usage:
#   ./build_tools/kokoro/gcp_ubuntu/simulate_kokoro.sh \
#     build_tools/kokoro/gcp_ubuntu/bazel/linux/x86-swiftshader/core/build_kokoro.sh
#
# Just does the part of the Kokoro setup that we care about and invokes the
# given build script.
# An optional second parameter can be used to specify a different repo to clone
# from. Especially useful for cloning the current git repo. If there's any local
# change, be sure to commit it before running this simulation.
#   ./build_tools/kokoro/gcp_ubuntu/simulate_kokoro.sh \
#     build_tools/kokoro/gcp_ubuntu/bazel/linux/x86-swiftshader/core/build_kokoro.sh \
#     "${PWD?}/.git"

set -x
set -e
set -o pipefail

RELATIVE_KOKORO_BUILD_SCRIPT="${1?}"
REPO_TO_CLONE="${2:-git@github.com:iree-org/iree.git}"

# Set up the temporary Kokoro directories
export KOKORO_ROOT="$(mktemp --directory --tmpdir kokoro-root-XXXXXX)"
mkdir -p "${KOKORO_ROOT?}/src/github"
export KOKORO_ARTIFACTS_DIR="${KOKORO_ROOT?}/src"
cd "${KOKORO_ARTIFACTS_DIR?}/github"

# Clone the repo
git clone "${REPO_TO_CLONE?}"

# The build script is assumed to be relative to the iree repo root.
KOKORO_BUILD_SCRIPT="${KOKORO_ARTIFACTS_DIR?}/github/iree/${RELATIVE_KOKORO_BUILD_SCRIPT?}"
chmod +x "${KOKORO_BUILD_SCRIPT?}"

# This is where Kokoro starts its execution.
cd "${KOKORO_ARTIFACTS_DIR?}"

# Run the actual script.
"${KOKORO_BUILD_SCRIPT?}"

# Clean up after ourselves.
rm -rf "${KOKORO_ROOT?}"

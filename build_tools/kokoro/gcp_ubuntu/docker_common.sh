# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Functions for setting up Docker containers to run on Kokoro

DOCKER_WORKDIR="${KOKORO_ARTIFACTS_DIR?}/github/iree"
DOCKER_TMPDIR="${KOKORO_ROOT?}"

source "${KOKORO_ARTIFACTS_DIR?}/github/iree/build_tools/docker/docker_run.sh"

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs docker configured for usage with GitHub Actions, translating GitHub
# Actions environment variables into generic ones and then invoking the generic
# docker_run script.

set -euo pipefail

export DOCKER_HOST_WORKDIR="${GITHUB_WORKSPACE}"
export DOCKER_HOST_TMPDIR="${RUNNER_TEMP}"

"${GITHUB_WORKSPACE}/build_tools/docker/docker_run.sh" "$@"

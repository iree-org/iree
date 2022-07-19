# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs docker configured for usage with Kokoro, translating Kokoro-specific
# environment variables into generic ones and then invoking `docker run` with
# some arguments to make it behave itself.

set -euo pipefail

export DOCKER_WORKDIR="${KOKORO_ARTIFACTS_DIR}/github/iree"
export DOCKER_TMPDIR="${KOKORO_ROOT}"

"${KOKORO_ARTIFACTS_DIR?}/github/iree/build_tools/docker/docker_run.sh" "$@"

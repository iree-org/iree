# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

source "${SCRIPT_DIR}/set_buildkite_env.sh"

bk run \
  --env=BUILDKITE_ORGANIZATION_SLUG="${BUILDKITE_ORGANIZATION_SLUG}" \
  --env=BUILDKITE_COMMIT="${BUILDKITE_COMMIT}" \
  --env=BUILDKITE_BRANCH="${BUILDKITE_BRANCH}" \
  --env=BUILDKITE_BUILD_AUTHOR="${BUILDKITE_BUILD_AUTHOR}" \
  --env=BUILDKITE_BUILD_AUTHOR_EMAIL="${BUILDKITE_BUILD_AUTHOR_EMAIL}" \
  $@

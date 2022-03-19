# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

export BUILDKITE_ORGANIZATION_SLUG="iree"
export BUILDKITE_COMMIT="$(git rev-parse HEAD)"
export BUILDKITE_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
export BUILDKITE_BUILD_AUTHOR="$(git log -n 1 --pretty="format:%aN" HEAD)"
export BUILDKITE_BUILD_AUTHOR_EMAIL="$(git log -n 1 --pretty="format:%aE" HEAD)"

bk run \
  --env=BUILDKITE_ORGANIZATION_SLUG="${BUILDKITE_ORGANIZATION_SLUG}" \
  --env=BUILDKITE_COMMIT="${BUILDKITE_COMMIT}" \
  --env=BUILDKITE_BRANCH="${BUILDKITE_BRANCH}" \
  --env=BUILDKITE_BUILD_AUTHOR="${BUILDKITE_BUILD_AUTHOR}" \
  --env=BUILDKITE_BUILD_AUTHOR_EMAIL="${BUILDKITE_BUILD_AUTHOR_EMAIL}" \
  $@

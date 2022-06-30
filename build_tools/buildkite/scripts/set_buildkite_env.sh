# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# *source* this file to set environment variables emulating the state when
# running in a Buildkite job.

export BUILDKITE_ORGANIZATION_SLUG="iree"
export BUILDKITE_PIPELINE_SLUG="local-test"
export BUILDKITE_BUILD_NUMBER=1
export BUILDKITE_REPO="https://github.com/iree-org/iree"
export BUILDKITE_COMMIT="$(git rev-parse HEAD)"
export BUILDKITE_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
export BUILDKITE_BUILD_AUTHOR="$(git log -n 1 --pretty="format:%aN" HEAD)"
export BUILDKITE_BUILD_AUTHOR_EMAIL="$(git log -n 1 --pretty="format:%aE" HEAD)"

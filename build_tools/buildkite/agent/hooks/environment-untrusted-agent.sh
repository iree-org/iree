# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

export BUILDKITE_ANNOTATION_CONTEXT="$${BUILDKITE_STEP_ID}"
export IREE_BUILDKITE_ACCESS_TOKEN="$(gcloud secrets versions access latest \
          --secret=iree-buildkite-presubmit-pipelines)"

#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script is intended to do the core build inside the Docker container once
# everything is configured. It aims to be relatively environment-agnostic, so
# it can be run in a container that's already configured or directly on a
# similar Linux box.

set -euo pipefail

cd integrations/tensorflow
# Ignore user-specific settings from the workspace rc file, but still capture
# environment settings from the system and home directory ones.
BAZEL_CMD=(bazel --noworkspace_rc --bazelrc=build_tools/bazel/iree-tf.bazelrc)
"${BAZEL_CMD[@]}" query //iree_tf_compiler/... | \
   xargs "${BAZEL_CMD[@]}" test
      --config=generic_clang \
      --test_tag_filters="-nokokoro" \
      --build_tag_filters="-nokokoro"

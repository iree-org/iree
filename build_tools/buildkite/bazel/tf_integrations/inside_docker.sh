#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cd integrations/tensorflow
BAZEL_CMD=(bazel --noworkspace_rc --bazelrc=build_tools/bazel/iree-tf.bazelrc)
"${BAZEL_CMD[@]?}" query //iree_tf_compiler/... | \
   xargs "${BAZEL_CMD[@]?}" test --config=generic_clang \
      --test_tag_filters="-nokokoro" \
      --build_tag_filters="-nokokoro"

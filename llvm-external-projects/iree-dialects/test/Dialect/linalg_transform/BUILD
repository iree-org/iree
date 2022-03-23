# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests for common transforms.

load("//build_tools/bazel:iree_lit_test.bzl", "iree_lit_test_suite")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "lit",
    srcs = ["bufferize.mlir"],
    # enforce_glob(
    #     # keep sorted
    #     [
    #       "bufferize.mlir",
    #       "double-tiling.mlir",
    #       "drop-schedule.mlir",
    #       "expert.mlir",
    #       "failure.mlir",
    #       "fuse.mlir",
    #       "generalize.mlir",
    #       "interchange.mlir",
    #       "invalid.mlir",
    #       "pad.mlir",
    #       "roundtrip.mlir",
    #       "scoped.mlir",
    #       "selective-targeting.mlir",
    #       "single-tiling-full-script.mlir",
    #       "tile-interchange.mlir",
    #       "tile.mlir",
    #       "vectorize.mlir",
    #     ],
    #     include = ["*.mlir"],
    #     # vectorize-transforms is a an MLIR file that specifies a
    #     # transformation, it needs to be included as data.
    #     exclude = ["vectorize-transforms.mlir"],
    # ),
    # data = ["vectorize-transforms.mlir"],
    tools = [
        "//llvm-external-projects/iree-dialects:iree-dialects-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

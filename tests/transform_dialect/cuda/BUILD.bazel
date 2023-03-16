# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests for end-to-end IREE support of entire models or their close derivatives.

load("//build_tools/bazel:build_defs.oss.bzl", "iree_cmake_extra_content")
load("//build_tools/bazel:iree_lit_test.bzl", "iree_lit_test_suite")

package(
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(NOT IREE_HAL_DRIVER_CUDA OR NOT IREE_TARGET_BACKEND_CUDA)
  return()
endif()
""",
    inline = True,
)

iree_lit_test_suite(
    name = "lit",
    srcs = [
        "mma.mlir",
        "mma_using_layout_analysis.mlir",
        "reduction.mlir",
        "reduction_eltwise.mlir",
        "reduction_v2.mlir",
        "reduction_v2_uneven.mlir",
        "reduction_v3.mlir",
        "softmax.mlir",
        "softmax_v2.mlir",
        # First few ops of softmax only, acts as a proxy example.
        "softmax_partial.mlir",
        "vecadd2d.mlir",
    ],
    cfg = "//tests:lit.cfg.py",
    # transform dialect spec files are MLIR files that specify a transformation,
    # they need to be included as data.
    data = [
        "mma_using_layout_analysis_codegen_spec.mlir",
        "reduction_codegen_spec.mlir",
        "reduction_eltwise_codegen_spec.mlir",
        "reduction_v2_codegen_spec.mlir",
        "reduction_v3_codegen_spec.mlir",
        "softmax_codegen_spec.mlir",
        "softmax_v2_codegen_spec.mlir",
        #
        # FIXME: This must be used with the custom dispatch region formation
        # because IREE's does not fuse the 6 ops softmax version even with
        # --iree-flow-enable-aggressive-fusion.
        #
        "softmax_dispatch_spec.mlir",
        # First few ops of softmax only, acts as a proxy example.
        "softmax_partial_codegen_spec.mlir",
        "vecadd2d_codegen_spec.mlir",
    ],
    tags = [
        # CUDA cuInit fails with sanitizer on.
        "noasan",
        "nomsan",
        "notsan",
        "noubsan",
        "requires-gpu-nvidia",
        "driver=cuda",
    ],
    tools = [
        "//tools:iree-compile",
        "//tools:iree-opt",
        "//tools:iree-run-module",
        "@llvm-project//llvm:FileCheck",
    ],
)

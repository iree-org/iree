# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests for common transforms.

load("//build_tools/bazel:iree_lit_test.bzl", "iree_lit_test_suite")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "lit",
    srcs = enforce_glob(
        [
            "broadcasting.mlir",
            "convert_mhlo_to_linalg_ext.mlir",
            "convert_collective_ops.mlir",
            "convert_complex_to_real.mlir",
            "convert_structural_types.mlir",
            "dynamic_shape.mlir",
            "fft.mlir",
            "flatten_tuples_in_cfg.mlir",
            "mhlo_to_linalg.mlir",
            "mhlo_to_mhlo_preprocessing.mlir",
            "mhlo_to_mhlo_preprocessing_canonicalize_dot_general.mlir",
            "mhlo_to_mhlo_scatter.mlir",
            "missing_legalizations.mlir",
            "transformation_pipeline.mlir",
            "verify_compiler_mhlo_input_legality.mlir",
        ],
        include = ["*.mlir"],
    ),
    cfg = "//compiler:lit.cfg.py",
    tools = [
        "//tools:iree-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

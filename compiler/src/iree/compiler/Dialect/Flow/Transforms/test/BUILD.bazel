# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
            "capture_dispatch_dynamic_dims.mlir",
            "cleanup_numeric_narrowing.mlir",
            "cleanup_tensor_shapes.mlir",
            "collapse_reduction.mlir",
            "conv1x1_to_matmul.mlir",
            "convert_region_to_workgroups.mlir",
            "deduplicate_executables.mlir",
            "detach_elementwise_from_named_ops.mlir",
            "dispatch_linalg_on_tensors.mlir",
            "collapse_linalg_generic_on_tensors.mlir",
            "dispatch_linalg_on_tensors_default.mlir",
            "dispatch_linalg_on_tensors_fusion_with_transpose.mlir",
            "dispatch_linalg_transform_dialect.mlir",
            "expand_tensor_shapes.mlir",
            "export_benchmark_funcs.mlir",
            "fusion_of_tensor_ops.mlir",
            "infer_numeric_narrowing.mlir",
            "initialize_empty_tensors.mlir",
            "inject_dispatch_tracing.mlir",
            "interchange_generic_ops.mlir",
            "interchange_transpose_generic_ops.mlir",
            "optimize_numerics.mlir",
            "outline_dispatch_regions.mlir",
            "raise_special_ops.mlir",
            "set_encoding.mlir",
            "strip_and_splat_constant_variables.mlir",
            "strip_signedness.mlir",
            "tensor_pad_to_tensor_insert_slice.mlir",
            "transform_dispatch_region_formation.mlir",
            "transformation_pipeline.mlir",
            "verify_input_ir.mlir",
        ],
        include = ["*.mlir"],
        # transform_dialect_dispatch_spec is a an MLIR file that specifies a
        # transformation, it needs to be included as data.
        exclude = [
            "transform_dialect_dispatch_spec.mlir",
        ],
    ),
    cfg = "//compiler:lit.cfg.py",
    data = ["transform_dialect_dispatch_spec.mlir"],
    tools = [
        "//tools:iree-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

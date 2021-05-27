# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//iree:lit_test.bzl", "iree_lit_test_suite")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "lit",
    srcs = enforce_glob(
        [
            "convert_to_flow_tensor_ops.mlir",
            "deduplicate_executables.mlir",
            "dispatch_linalg_on_tensors.mlir",
            "dispatch_linalg_on_tensors_elementwise.mlir",
            "dispatch_linalg_on_tensors_fusion.mlir",
            "expand_variable_dynamic_dims.mlir",
            "export_benchmark_funcs.mlir",
            "form_streams.mlir",
            "hlo_to_hlo_preprocessing.mlir",
            "hlo_to_hlo_preprocessing_canoncalize_dot_general.mlir",
            "hlo_to_hlo_preprocessing_extract_pad_from_conv.mlir",
            "hoist_unstreamable_ops.mlir",
            "inject_dispatch_tracing.mlir",
            "legalize_input_types.mlir",
            "outline_dispatch_regions.mlir",
            "outline_large_constants.mlir",
            "pre_partitioning_conversion.mlir",
            "promote_tensor_loads.mlir",
            "strip_and_splat_constant_variables.mlir",
            "transformation.mlir",
            "verify_compiler_input_legality.mlir",
        ],
        include = ["*.mlir"],
    ),
    data = [
        "//iree/tools:IreeFileCheck",
        "//iree/tools:iree-opt",
    ],
)

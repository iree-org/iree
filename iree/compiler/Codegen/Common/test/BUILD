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
    srcs = enforce_glob(
        [
            "affinemin_canonicalization.mlir",
            "bufferize_copy_only_dispatches.mlir",
            "canonicalize_interface_load_store.mlir",
            "convert_to_destination_passing_style.mlir",
            "dead_alloc.mlir",
            "flatten_memref_subspan.mlir",
            "fold_affine_min_in_distributed_loops.mlir",
            "fold_tensor_extract_op.mlir",
            "forop_canonicalization.mlir",
            "insert_distribution_info.mlir",
            "iree_comprehensive_bufferize.mlir",
            "linalg_bufferize.mlir",
            "remove_dead_allocs.mlir",
            "remove_trivial_loops.mlir",
            "rewrite_linalg_destructive_updates.mlir",
            "swizzle_workgroup.mlir",
            "tile_and_distribute_to_workgroups.mlir",
            "transpose_canonicalization.mlir",
            "type_propagation.mlir",
            "vectorize_linalg_conv.mlir",
            "vectorize_linalg_mmt4d.mlir",
        ],
        include = ["*.mlir"],
    ),
    tools = [
        "//iree/tools:iree-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

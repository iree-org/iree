# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests for common transforms.

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
            "affinemin_canonicalization.mlir",
            "canonicalize_interface_load_store.mlir",
            "dead_alloc.mlir",
            "f32Tof16.mlir",
            "flatten_memref_subspan.mlir",
            "fold_affine_min_in_distributed_loops.mlir",
            "fold_tensor_extract_op.mlir",
            "forop_canonicalization.mlir",
            "iree_comprehensive_bufferize.mlir",
            "linalg_bufferize.mlir",
            "remove_dead_allocs.mlir",
            "transpose_canonicalization.mlir",
            "vectorize_linalg_conv.mlir",
            "vectorize_linalg_mmt4d.mlir",
        ],
        include = ["*.mlir"],
    ),
    data = [
        "//iree/tools:IreeFileCheck",
        "//iree/tools:iree-opt",
    ],
)

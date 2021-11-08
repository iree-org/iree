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
            "config_adreno_conv.mlir",
            "config_adreno_matmul.mlir",
            "config_default_linalg_ext_ops.mlir",
            "config_default_linalg_ops.mlir",
            "config_default_matmul.mlir",
            "config_mali_conv.mlir",
            "config_mali_matmul.mlir",
            "config_nvidia_matmul_cooperative_ops.mlir",
            "convert_to_spirv.mlir",
            "pipeline_matmul_cooperative_ops.mlir",
            "pipeline_matmul_vectorization.mlir",
            "remove_one_trip_tiled_loop.mlir",
            "tile_and_distribute_scatter.mlir",
            "tile_and_distribute_sort.mlir",
            "tile_and_vectorize.mlir",
            "tile_and_vectorize_batch_matmul.mlir",
            "tile_and_vectorize_conv.mlir",
            "tile_and_vectorize_matmul.mlir",
            "tile_and_vectorize_to_cooperative_ops.mlir",
            "vector_to_cooperative_matrix.mlir",
            "vectorize_copy_to_workgroup_memory.mlir",
            "vectorize_elementwise_ops.mlir",
            "vectorize_matmul.mlir",
            "vectorize_load_store.mlir",
        ],
        include = ["*.mlir"],
        # TODO(b/203528778) reenable
        exclude = [
            "promote_workgroup_memory.mlir",
        ],
    ),
    data = [
        "//iree/tools:IreeFileCheck",
        "//iree/tools:iree-opt",
    ],
)

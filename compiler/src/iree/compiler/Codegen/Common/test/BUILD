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
            "affinemin_canonicalization.mlir",
            "bufferize_copy_only_dispatches.mlir",
            "canonicalize_interface_load_store.mlir",
            "convert_to_destination_passing_style.mlir",
            "dead_alloc.mlir",
            "decompose_linalg_generic.mlir",
            "distribute_gpu_shared_memory.mlir",
            "eliminate_empty_tensors.mlir",
            "erase_hal_descriptor_type.mlir",
            "fixup_subspan_with_offsets.mlir",
            "flatten_memref_subspan.mlir",
            "fold_affine_min_in_distributed_loops.mlir",
            "fold_tensor_extract_op.mlir",
            "forop_canonicalization.mlir",
            "gpu_pipeline.mlir",
            "gpu_vectorization.mlir",
            "hoist_statically_bound_allocations.mlir",
            "iree_comprehensive_bufferize.mlir",
            "lower_ukernel_to_calls.mlir",
            "pad_dynamic_alloc.mlir",
            "rematerialize_parallel_ops.mlir",
            "reduce_bank_conflicts.mlir",
            "reductions.mlir",
            "remove_dead_allocs.mlir",
            "remove_trivial_loops.mlir",
            "repeated_matcher_use.mlir",
            "swizzle_workgroup.mlir",
            "test_partitionable_loops_interface.mlir",
            "tile_and_distribute_to_workgroups.mlir",
            "tile_reduction.mlir",
            "transform_buffer_opt.mlir",
            "transform_dialect_apply_pattern_op.mlir",
            "transform_ops_invalid.mlir",
            "transpose_canonicalization.mlir",
            "type_propagation.mlir",
            "vectorize_pack_unpack_ops.mlir",
            "vectorize_tensor_pad.mlir",
            "warp_reduction.mlir",
            "workgroup_specialization.mlir",
        ],
        include = ["*.mlir"],
        exclude = [
            "reductions_codegen_spec.mlir",
            "reductions_match_spec.mlir",
        ],
    ),
    cfg = "//compiler:lit.cfg.py",
    # transform dialect spec files are MLIR files that specify a transformation,
    # they need to be included as data.
    data = [
        "reductions_codegen_spec.mlir",
        "reductions_match_spec.mlir",
    ],
    tools = [
        "//tools:iree-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

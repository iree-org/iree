// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Explicit numerical comparison profile for SmolLM2-135M on SPIR-V targets.
// Distribute output elements and keep each dot product in a scalar FMA loop.
// This matches the accumulation order of the pinned Torch reference in the
// recorded validation environment; it is not a general accuracy or speed claim.
//
// Match only f32 tensor MMT projections with reduction width 576 or 1536 and
// an output column count proven divisible by 32. Other operations and existing
// user configurations are left intact. Select this file with
// --iree-codegen-transform-dialect-library; it does not change compiler defaults.

module @smollm2_spirv_f32 attributes {transform.with_named_sequence} {
  transform.named_sequence @apply_config(%op: !transform.any_op {transform.readonly}) {
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 32], [1, 1], [0, 0, 1]]>,
      translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<BaseVectorize> workgroup_size = [32, 1, 1]>
    > -> !transform.any_param
    transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @match_hidden_projection(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.iree.match.has_no_lowering_config %op : !transform.any_op
    %batch, %m, %n, %k = transform.iree.match.contraction %op,
      lhs_type = f32, rhs_type = f32, output_type = f32,
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %k, [576] : !transform.param<i64>
    %lhs = transform.get_operand %op[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<?x576xf32> : !transform.any_value
    %rhs = transform.get_operand %op[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<?x576xf32> : !transform.any_value
    transform.iree.match.dim_is_multiple_of %rhs[0], 32 : !transform.any_value
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @match_intermediate_projection(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.iree.match.has_no_lowering_config %op : !transform.any_op
    %batch, %m, %n, %k = transform.iree.match.contraction %op,
      lhs_type = f32, rhs_type = f32, output_type = f32,
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (n, k)>, affine_map<(m, n, k) -> (m, n)>] : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %k, [1536] : !transform.param<i64>
    %lhs = transform.get_operand %op[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<?x1536xf32> : !transform.any_value
    %rhs = transform.get_operand %op[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<?x1536xf32> : !transform.any_value
    transform.iree.match.dim_is_multiple_of %rhs[0], 32 : !transform.any_value
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @__kernel_config(%function: !transform.any_op {transform.consumed}) -> !transform.any_op attributes {iree_codegen.tuning_spec_entrypoint} {
    %result = transform.foreach_match in %function
      @match_hidden_projection -> @apply_config,
      @match_intermediate_projection -> @apply_config
      : (!transform.any_op) -> !transform.any_op
    transform.yield %result : !transform.any_op
  }
}

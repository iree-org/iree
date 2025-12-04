// RUN: iree-opt %s

// This is just an initial tuning spec for gfx942 and is not intended for
// production use.
// TODO(https://github.com/iree-org/iree/issues/19214): Add missing
// configurations to this spec.

module @iree_default_tuning_spec_gfx942 attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {

// ============================================================
// * Tuning Configurations Start *
// ============================================================

transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly},
                                        %config: !transform.any_param {transform.readonly}) {
  transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
  // Add a dummy unit attribute to be sure that the tuning spec applied.
  // Otherwise it would be difficult to tell if the lowering config attribute
  // comes from our tuning spec or if the compiler heuristic happened to produce
  // the same config as this script.
  transform.annotate %op "__tuning_spec_applied__" : !transform.any_op
  transform.yield
}

transform.named_sequence @apply_attn_op_config(%attention: !transform.any_op {transform.readonly},
                                                %config: !transform.any_param {transform.readonly},
                                                %decomposition_config: !transform.any_param {transform.readonly}) {
  transform.annotate %attention "compilation_info" = %config : !transform.any_op, !transform.any_param
  transform.annotate %attention "decomposition_config" = %decomposition_config : !transform.any_op, !transform.any_param
  transform.annotate %attention "__tuning_spec_applied__" : !transform.any_op
  transform.yield
}

transform.named_sequence
@match_attention_2x10x4096x64x64x64_f16(%attention: !transform.any_op {transform.readonly})
  -> (!transform.any_op, !transform.any_param, !transform.any_param) {
  transform.iree.match.has_no_lowering_config %attention : !transform.any_op

  %batch, %m, %k1, %k2, %n =
    transform.iree.match.attention %attention,
      query_type = f16, key_type = f16, value_type = f16, output_type = f16,
      indexing_maps = [
        affine_map<(B0, B1, M, N, K1, K2) -> (B0, B1, M, K1)>,
        affine_map<(B0, B1, M, N, K1, K2) -> (B0, B1, K2, K1)>,
        affine_map<(B0, B1, M, N, K1, K2) -> (B0, B1, N, K2)>,
        affine_map<(B0, B1, M, N, K1, K2) -> ()>,
        affine_map<(B0, B1, M, N, K1, K2) -> (B0, B1, M, N)>
      ] : !transform.any_op -> !transform.param<i64>

  %query = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
  %key = transform.get_operand %attention[1] : (!transform.any_op) -> !transform.any_value
  %value = transform.get_operand %attention[2] : (!transform.any_op) -> !transform.any_value

  transform.iree.match.cast_compatible_type %query = tensor<?x?x?x?xf16> : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %query[2], 128 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %query[3], 16 : !transform.any_value
  transform.iree.match.cast_compatible_type %key = tensor<?x?x64x64xf16> : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %key[2], 64 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %key[3], 16 : !transform.any_value
  transform.iree.match.cast_compatible_type %value = tensor<?x?x64x64xf16> : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %value[2], 16 : !transform.any_value
  transform.iree.match.dim_is_multiple_of  %value[3], 64 : !transform.any_value

  // `amdgpu-waves-per-eu`:
  // The gfx942 GPU attention implementation uses a high number of registers.
  // Setting this flag instructs the compiler to be less conservative in register allocation,
  // leading to better performance.

  // `denormal-fp-math-f32`:
  // Disables denormal flushing for `exp2/exp` operations, reducing the number of instructions
  // required for exp/exp2.
  %config = transform.param.constant #iree_codegen.compilation_info<
          lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 1, 128, 0, 0, 0], reduction=[0, 0, 0, 0, 0, 64], promote_operands = [1, 2]}>,
          translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                                            workgroup_size = [256]
                                                            subgroup_size = 64 ,
            {llvm_func_attrs = { "amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign" }}>>
  -> !transform.any_param

  // `promote_operands = [1]`:
  // - Only `K` and `V` tensors are promoted to shared memory.
  // - `Q` is not promoted since the `QK` matrix multiplication uses VMFMA instructions,
  //   which operate efficiently with `vector<8xf16>` from global memory.
  %decomposition_config = transform.param.constant {
    qk_attrs = {attention_qk_matmul,
                lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.virtual_mma_layout<VMFMA_F32_32x32x16_F16>,
                                                              subgroup_basis = [[1, 1, 4, 1, 1, 1], [0, 1, 2, 4, 5]], promote_operands = [1] }>},
    pv_attrs = {attention_pv_matmul,
                lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
                                                              subgroup_basis = [[1, 1, 4, 1, 1, 1], [0, 1, 2, 3, 5]], promote_operands = [1] }>}
  } -> !transform.any_param

  transform.yield %attention, %config, %decomposition_config : !transform.any_op, !transform.any_param, !transform.any_param
}

transform.named_sequence
@match_mmt_2048x1280x5120_f16_f16_f32(%matmul: !transform.any_op {transform.readonly})
  -> (!transform.any_op, !transform.any_param) {
  transform.iree.match.has_no_lowering_config %matmul : !transform.any_op

  %batch, %m, %n, %k = transform.iree.match.contraction %matmul,
    lhs_type = f16, rhs_type = f16, output_type = f32,
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>] : !transform.any_op -> !transform.param<i64>
  transform.iree.match.dims_equal %m, [2048] : !transform.param<i64>
  transform.iree.match.dims_equal %n, [1280] : !transform.param<i64>
  transform.iree.match.dims_equal %k, [5120] : !transform.param<i64>
  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                 mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                                 subgroup_basis = [[2, 2, 1], [0, 1, 2]],
                                                 reduction = [0, 0, 64],
                                                 workgroup = [64, 128, 0]}>,
    translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
      workgroup_size = [256, 1, 1] subgroup_size = 64,
      {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>}>
  > -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence
@__kernel_config(%variant_op: !transform.any_op {transform.consumed}) -> !transform.any_op
  attributes { iree_codegen.tuning_spec_entrypoint } {
  %res = transform.foreach_match in %variant_op
    // Expected speedup: 1.22x.
    @match_attention_2x10x4096x64x64x64_f16 -> @apply_attn_op_config,
    @match_mmt_2048x1280x5120_f16_f16_f32 -> @apply_op_config
    : (!transform.any_op) -> !transform.any_op
  transform.yield %res : !transform.any_op
}

}

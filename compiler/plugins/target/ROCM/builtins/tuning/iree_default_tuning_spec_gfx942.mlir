// RUN: iree-opt %s

// This is just an initial tuning spec for gfx942 and is not intended for
// production use.
// TODO(https://github.com/iree-org/iree/issues/19214): Add missing
// configurations to this spec.

module @iree_default_tuning_spec_gfx942 attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {

transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly},
                                        %config: !transform.any_param {transform.readonly}) {
  // transform.print %op {name="Apply on"} : !transform.any_op
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

// ============================================================
// * Tuning Configurations Start *
// ============================================================

transform.named_sequence @match_attention_f16(%root: !transform.any_op {transform.readonly})
  -> !transform.any_op {
  transform.match.operation_name %root ["iree_linalg_ext.attention"] : !transform.any_op
  %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
    ^bb0(%query: tensor<?x?x?x?xf16>,
         %key: tensor<?x?x?x?xf16>,
         %value: tensor<?x?x?x?xf16>,
         %softmax_scale: f16,
         %out: tensor<?x?x?x?xf16>):

      %attn = iree_linalg_ext.attention {indexing_maps = [
                                          affine_map<(B0, B1, M, N, K1, K2) -> (B0, B1, M, K1)>,
                                          affine_map<(B0, B1, M, N, K1, K2) -> (B0, B1, K2, K1)>,
                                          affine_map<(B0, B1, M, N, K1, K2) -> (B0, B1, N, K2)>,
                                          affine_map<(B0, B1, M, N, K1, K2) -> ()>,
                                          affine_map<(B0, B1, M, N, K1, K2) -> (B0, B1, M, N)>]}
        ins(%query, %key, %value, %softmax_scale :
            tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, f16)
        outs(%out : tensor<?x?x?x?xf16>){
          ^bb0(%arg0: f32):
            iree_linalg_ext.yield %arg0 : f32
        } -> tensor<?x?x?x?xf16>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)

  transform.yield %root : !transform.any_op
}

transform.named_sequence
@match_attention_2x10x4096x64x64x64_f16(%attention: !transform.any_op {transform.readonly})
  -> (!transform.any_op, !transform.any_param, !transform.any_param) {

  %matched = transform.include @match_attention_f16 failures(propagate) (%attention)
    : (!transform.any_op) -> !transform.any_op

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
                lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.virtual_mma_layout<intrinsic = VMFMA_F32_32x32x16_F16>,
                                                              subgroup_m_count = 4, subgroup_n_count = 1, promote_operands = [1] }>},
    pv_attrs = {attention_pv_matmul,
                lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
                                                              subgroup_m_count = 4, subgroup_n_count = 1, promote_operands = [1] }>}
  } -> !transform.any_param

  transform.yield %attention, %config, %decomposition_config : !transform.any_op, !transform.any_param, !transform.any_param
}

transform.named_sequence @match_mmt_f16_f16_f32(%root: !transform.any_op {transform.readonly}) -> !transform.any_op {
  transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
  // transform.print %root {name = "Generic"} : !transform.any_op
  %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
    ^bb0(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>, %out: tensor<?x?xf32>):
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                          affine_map<(d0, d1, d2) -> (d1, d2)>,
                                          affine_map<(d0, d1, d2) -> (d0, d1)>],
                          iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs : tensor<?x?xf16>, tensor<?x?xf16>) outs(%out : tensor<?x?xf32>) {
      ^bb0(%in: f16, %in_0: f16, %acc: f32):
        %8 = arith.extf %in : f16 to f32
        %9 = arith.extf %in_0 : f16 to f32
        %10 = arith.mulf %8, %9 : f32
        %11 = arith.addf %acc, %10 : f32
        linalg.yield %11 : f32
      } -> tensor<?x?xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  transform.yield %root : !transform.any_op
}

transform.named_sequence
@match_mmt_2048x1280x5120_f16_f16_f32(%matmul: !transform.any_op {transform.readonly})
  -> (!transform.any_op, !transform.any_param) {
  %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul)
    : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
  transform.iree.match.cast_compatible_type %lhs = tensor<2048x5120xf16> : !transform.any_value
  transform.iree.match.cast_compatible_type %rhs = tensor<1280x5120xf16> : !transform.any_value
  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                 mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                                 subgroup_m_count = 2, subgroup_n_count = 2,
                                                 reduction = [0, 0, 64],
                                                 workgroup = [64, 128, 0]}>,
    translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
      workgroup_size = [256, 1, 1] subgroup_size = 64,
      {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>}>
  > -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence
@__kernel_config(%variant_op: !transform.any_op {transform.consumed}) -> !transform.any_op
  attributes { iree_codegen.tuning_spec_entrypoint } {
  %res = transform.foreach_match in %variant_op
    // Expected speedup: 1.22x.
    @match_attention_2x10x4096x64x64x64_f16 -> @apply_attn_op_config
    , @match_mmt_2048x1280x5120_f16_f16_f32 -> @apply_op_config
    : (!transform.any_op) -> !transform.any_op
  transform.yield %res : !transform.any_op
}

}

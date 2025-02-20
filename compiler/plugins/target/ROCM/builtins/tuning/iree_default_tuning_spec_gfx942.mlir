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

transform.named_sequence @match_attention_f16(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param, !transform.any_param) {
    transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
    %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<?x?x?x?xf16> : !transform.any_value

    %config = transform.param.constant #iree_codegen.compilation_info<
            lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 1, 128, 0, 0, 0], reduction=[0, 0, 0, 0, 0, 64], promote_operands = [1, 2]}>,
            translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                                              workgroup_size = [64, 4]
                                                              subgroup_size = 64 ,
              {llvm_func_attrs = { "amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign" }}>>
    -> !transform.any_param

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
    @match_attention_f16 -> @apply_attn_op_config
    , @match_mmt_2048x1280x5120_f16_f16_f32 -> @apply_op_config
    : (!transform.any_op) -> !transform.any_op
  transform.yield %res : !transform.any_op
}

}

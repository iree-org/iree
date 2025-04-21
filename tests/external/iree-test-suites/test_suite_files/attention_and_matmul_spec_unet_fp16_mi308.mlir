module attributes { transform.with_named_sequence } {
//===----------------------------------------------------------------------===//
// Tuning infra
//===----------------------------------------------------------------------===//

transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly},
                                          %config: !transform.any_param {transform.readonly}) {
  transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
  // transform.print %op {name = "Applied"} : !transform.any_op
  transform.yield
}

transform.named_sequence @apply_attn_op_config(%attention: !transform.any_op {transform.readonly},
                                                %config: !transform.any_param {transform.readonly},
                                                %decomposition_config: !transform.any_param {transform.readonly}) {
  transform.annotate %attention "compilation_info" = %config : !transform.any_op, !transform.any_param
  transform.annotate %attention "decomposition_config" = %decomposition_config : !transform.any_op, !transform.any_param
  // transform.print %attention {name = "Applied attention config"} : !transform.any_op
  transform.yield
}

transform.named_sequence @match_attention_f16(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param, !transform.any_param) {
  transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
  %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
  transform.iree.match.cast_compatible_type %in0 = tensor<?x?x?x?xf16> : !transform.any_value

  %config = transform.param.constant #iree_codegen.compilation_info<
          lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 1, 64, 0, 0, 0], reduction=[0, 0, 0, 0, 0, 64], promote_operands = [1, 2]}>,
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

transform.named_sequence @match_mmt_f16_f16_f32(%root: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
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
        %18 = arith.extf %in : f16 to f32
        %19 = arith.extf %in_0 : f16 to f32
        %20 = arith.mulf %18, %19 : f32
        %21 = arith.addf %acc, %20 : f32
        linalg.yield %21 : f32
      } -> tensor<?x?xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  transform.yield %root : !transform.any_op
}

// TUNING_SPEC_BEGIN DO NOT REMOVE

//===----------------------------------------------------------------------===//
// Matmul tuning
//===----------------------------------------------------------------------===//

transform.named_sequence @match_mmt_1920x10240x1280(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
  transform.iree.match.cast_compatible_type %lhs = tensor<1920x1280xf16> : !transform.any_value
  transform.iree.match.cast_compatible_type %rhs = tensor<10240x1280xf16> : !transform.any_value
  %config = transform.param.constant #iree_codegen.compilation_info<
  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                                subgroup_m_count = 4, subgroup_n_count = 2,
                                                reduction = [0, 0, 32],
                                                workgroup = [128, 128, 0]}>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
    workgroup_size = [128, 4, 1] subgroup_size = 64,
    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>,
     llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}
    }>> -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence @match_mmt_1920x1280x1280(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
  transform.iree.match.cast_compatible_type %lhs = tensor<1920x1280xf16> : !transform.any_value
  transform.iree.match.cast_compatible_type %rhs = tensor<1280x1280xf16> : !transform.any_value
  %config = transform.param.constant #iree_codegen.compilation_info<
  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                                subgroup_m_count = 4, subgroup_n_count = 2,
                                                reduction = [0, 0, 32],
                                                workgroup = [128, 128, 0]}>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
    workgroup_size = [128, 4, 1] subgroup_size = 64,
    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>,
     llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}
    }>> -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence @match_mmt_1920x1280x5120(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
  transform.iree.match.cast_compatible_type %lhs = tensor<1920x5120xf16> : !transform.any_value
  transform.iree.match.cast_compatible_type %rhs = tensor<1280x5120xf16> : !transform.any_value
  %config = transform.param.constant #iree_codegen.compilation_info<
  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                                subgroup_m_count = 4, subgroup_n_count = 2,
                                                reduction = [0, 0, 32],
                                                workgroup = [128, 128, 0]}>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
    workgroup_size = [128, 4, 1] subgroup_size = 64,
    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>,
     llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}
    }>> -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence @match_mmt_7680x5120x640(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
  transform.iree.match.cast_compatible_type %lhs = tensor<7680x640xf16> : !transform.any_value
  transform.iree.match.cast_compatible_type %rhs = tensor<5120x640xf16> : !transform.any_value
  %config = transform.param.constant #iree_codegen.compilation_info<
  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                                subgroup_m_count = 2, subgroup_n_count = 4,
                                                reduction = [0, 0, 32],
                                                workgroup = [128, 256, 0]}>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
    workgroup_size = [256, 2, 1] subgroup_size = 64,
    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>,
     llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}
    }>> -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence @match_mmt_128x1280x2048(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
  transform.iree.match.cast_compatible_type %lhs = tensor<1280x2048xf16> : !transform.any_value
  transform.iree.match.cast_compatible_type %rhs = tensor<1280x2048xf16> : !transform.any_value
  %config = transform.param.constant #iree_codegen.compilation_info<
  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                                subgroup_m_count = 2, subgroup_n_count = 1,
                                                reduction = [0, 0, 128],
                                                workgroup = [64, 16, 0]}>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
    workgroup_size = [64, 2, 1] subgroup_size = 64,
    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>,
     llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}
    }>> -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence @match_mmt_7680x640x640(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
  transform.iree.match.cast_compatible_type %lhs = tensor<7680x640xf16> : !transform.any_value
  transform.iree.match.cast_compatible_type %rhs = tensor<640x640xf16> : !transform.any_value
  %config = transform.param.constant #iree_codegen.compilation_info<
  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
                                                subgroup_m_count = 1, subgroup_n_count = 4,
                                                reduction = [0, 0, 32],
                                                workgroup = [256, 128, 0]}>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
    workgroup_size = [256, 1, 1] subgroup_size = 64,
    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>,
     llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}
    }>> -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence @match_mmt_7680x640x2560(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
  transform.iree.match.cast_compatible_type %lhs = tensor<7680x2560xf16> : !transform.any_value
  transform.iree.match.cast_compatible_type %rhs = tensor<640x2560xf16> : !transform.any_value
  %config = transform.param.constant #iree_codegen.compilation_info<
  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
                                                subgroup_m_count = 4, subgroup_n_count = 2,
                                                reduction = [0, 0, 32],
                                                workgroup = [256, 128, 0]}>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
    workgroup_size = [128, 4, 1] subgroup_size = 64,
    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>,
     llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}
    }>> -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

//===----------------------------------------------------------------------===//
// Convolution tuning
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Batch matmul tuning
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Broadcast rhs mmt tuning
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Contraction tuning
//===----------------------------------------------------------------------===//

// TUNING_SPEC_END DO NOT REMOVE

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

  transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %variant_op
        @match_attention_f16 -> @apply_attn_op_config

        // TUNING_MATCH_BEGIN DO NOT REMOVE

        // MMT.
        , @match_mmt_1920x10240x1280 -> @apply_op_config
        , @match_mmt_1920x1280x1280 -> @apply_op_config
        , @match_mmt_1920x1280x5120 -> @apply_op_config
        , @match_mmt_7680x5120x640 -> @apply_op_config
        , @match_mmt_128x1280x2048 -> @apply_op_config
        , @match_mmt_7680x640x640 -> @apply_op_config
        , @match_mmt_7680x640x2560 -> @apply_op_config

        // TUNING_MATCH_END DO NOT REMOVE
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} ////  module

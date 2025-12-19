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
  transform.iree.match.has_no_lowering_config %attention : !transform.any_op

  transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
  %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
  transform.iree.match.cast_compatible_type %in0 = tensor<?x?x?x?xf16> : !transform.any_value

  %config = transform.param.constant #iree_codegen.compilation_info<
          lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 1, 64, 0, 0, 0], reduction=[0, 0, 0, 0, 0, 64], promote_operands = [1, 2]}>,
          translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                                            workgroup_size = [64, 2]
                                                            subgroup_size = 64 ,
            {llvm_func_attrs = { "amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign" }}>>
  -> !transform.any_param

  %decomposition_config = transform.param.constant {
    qk_attrs = {attention_qk_matmul,
                lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.virtual_mma_layout<VMFMA_F32_32x32x16_F16>,
                                                              subgroup_basis = [[1, 1, 2, 1, 1, 1], [0, 1, 2, 4, 5]], promote_operands = [1] }>},
    pv_attrs = {attention_pv_matmul,
                lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
                                                              subgroup_basis = [[1, 1, 2, 1, 1, 1], [0, 1, 2, 3, 5]], promote_operands = [1] }>}
  } -> !transform.any_param

  transform.yield %attention, %config, %decomposition_config : !transform.any_op, !transform.any_param, !transform.any_param
}

// TUNING_SPEC_BEGIN DO NOT REMOVE

//===----------------------------------------------------------------------===//
// Matmul tuning
//===----------------------------------------------------------------------===//

transform.named_sequence @match_mmt_1920x10240x1280(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  transform.iree.match.has_no_lowering_config %matmul : !transform.any_op

  %batch, %m, %n, %k = transform.iree.match.contraction %matmul,
    lhs_type = f16, rhs_type = f16, output_type = f32,
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>] : !transform.any_op -> !transform.param<i64>
  transform.iree.match.dims_equal %m, [1920] : !transform.param<i64>
  transform.iree.match.dims_equal %n, [10240] : !transform.param<i64>
  transform.iree.match.dims_equal %k, [1280] : !transform.param<i64>
  %config = transform.param.constant #iree_codegen.compilation_info<
  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                                subgroup_basis = [[4, 2, 1], [0, 1, 2]],
                                                reduction = [0, 0, 32],
                                                workgroup = [128, 128, 0]}>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
    workgroup_size = [128, 4, 1] subgroup_size = 64,
    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>,
     llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}
    }>> -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence @match_mmt_1920x1280x1280(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  transform.iree.match.has_no_lowering_config %matmul : !transform.any_op

  %batch, %m, %n, %k = transform.iree.match.contraction %matmul,
    lhs_type = f16, rhs_type = f16, output_type = f32,
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>] : !transform.any_op -> !transform.param<i64>
  transform.iree.match.dims_equal %m, [1920] : !transform.param<i64>
  transform.iree.match.dims_equal %n, [1280] : !transform.param<i64>
  transform.iree.match.dims_equal %k, [1280] : !transform.param<i64>
  %config = transform.param.constant #iree_codegen.compilation_info<
  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                                subgroup_basis = [[4, 2, 1], [0, 1, 2]],
                                                reduction = [0, 0, 32],
                                                workgroup = [128, 128, 0]}>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
    workgroup_size = [128, 4, 1] subgroup_size = 64,
    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>,
     llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}
    }>> -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence @match_mmt_1920x1280x5120(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  transform.iree.match.has_no_lowering_config %matmul : !transform.any_op

  %batch, %m, %n, %k = transform.iree.match.contraction %matmul,
    lhs_type = f16, rhs_type = f16, output_type = f32,
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>] : !transform.any_op -> !transform.param<i64>
  transform.iree.match.dims_equal %m, [1920] : !transform.param<i64>
  transform.iree.match.dims_equal %n, [1280] : !transform.param<i64>
  transform.iree.match.dims_equal %k, [5120] : !transform.param<i64>
  %config = transform.param.constant #iree_codegen.compilation_info<
  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                                subgroup_basis = [[4, 2, 1], [0, 1, 2]],
                                                reduction = [0, 0, 32],
                                                workgroup = [128, 128, 0]}>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
    workgroup_size = [128, 4, 1] subgroup_size = 64,
    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>,
     llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}
    }>> -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence @match_mmt_7680x5120x640(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  transform.iree.match.has_no_lowering_config %matmul : !transform.any_op

  %batch, %m, %n, %k = transform.iree.match.contraction %matmul,
    lhs_type = f16, rhs_type = f16, output_type = f32,
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>] : !transform.any_op -> !transform.param<i64>
  transform.iree.match.dims_equal %m, [7680] : !transform.param<i64>
  transform.iree.match.dims_equal %n, [5120] : !transform.param<i64>
  transform.iree.match.dims_equal %k, [640] : !transform.param<i64>
  %config = transform.param.constant #iree_codegen.compilation_info<
  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                                subgroup_basis = [[2, 4, 1], [0, 1, 2]],
                                                reduction = [0, 0, 32],
                                                workgroup = [128, 256, 0]}>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
    workgroup_size = [256, 2, 1] subgroup_size = 64,
    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>,
     llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}
    }>> -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence @match_mmt_128x1280x2048(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  transform.iree.match.has_no_lowering_config %matmul : !transform.any_op

  %batch, %m, %n, %k = transform.iree.match.contraction %matmul,
    lhs_type = f16, rhs_type = f16, output_type = f32,
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>] : !transform.any_op -> !transform.param<i64>
  transform.iree.match.dims_equal %m, [128] : !transform.param<i64>
  transform.iree.match.dims_equal %n, [1280] : !transform.param<i64>
  transform.iree.match.dims_equal %k, [2048] : !transform.param<i64>
  %config = transform.param.constant #iree_codegen.compilation_info<
  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                                subgroup_basis = [[2, 1, 1], [0, 1, 2]],
                                                reduction = [0, 0, 128],
                                                workgroup = [64, 16, 0]}>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
    workgroup_size = [64, 2, 1] subgroup_size = 64,
    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>,
     llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}
    }>> -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence @match_mmt_7680x640x640(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  transform.iree.match.has_no_lowering_config %matmul : !transform.any_op

  %batch, %m, %n, %k = transform.iree.match.contraction %matmul,
    lhs_type = f16, rhs_type = f16, output_type = f32,
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>] : !transform.any_op -> !transform.param<i64>
  transform.iree.match.dims_equal %m, [7680] : !transform.param<i64>
  transform.iree.match.dims_equal %n, [640] : !transform.param<i64>
  transform.iree.match.dims_equal %k, [640] : !transform.param<i64>
  %config = transform.param.constant #iree_codegen.compilation_info<
  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
                                                subgroup_basis = [[1, 4, 1], [0, 1, 2]],
                                                reduction = [0, 0, 32],
                                                workgroup = [256, 128, 0]}>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
    workgroup_size = [256, 1, 1] subgroup_size = 64,
    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>,
     llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}
    }>> -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence @match_mmt_7680x640x2560(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  transform.iree.match.has_no_lowering_config %matmul : !transform.any_op

  %batch, %m, %n, %k = transform.iree.match.contraction %matmul,
    lhs_type = f16, rhs_type = f16, output_type = f32,
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>] : !transform.any_op -> !transform.param<i64>
  transform.iree.match.dims_equal %m, [7680] : !transform.param<i64>
  transform.iree.match.dims_equal %n, [640] : !transform.param<i64>
  transform.iree.match.dims_equal %k, [2560] : !transform.param<i64>
  %config = transform.param.constant #iree_codegen.compilation_info<
  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
                                                subgroup_basis = [[4, 2, 1], [0, 1, 2]],
                                                reduction = [0, 0, 32],
                                                workgroup = [256, 128, 0]}>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
    workgroup_size = [128, 4, 1] subgroup_size = 64,
    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>,
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

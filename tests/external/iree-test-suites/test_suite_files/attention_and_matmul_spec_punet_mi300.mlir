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

//===----------------------------------------------------------------------===//
// Attention tuning
//===----------------------------------------------------------------------===//

transform.named_sequence @match_attention_f16(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %attention : !transform.any_op

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
                  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.virtual_mma_layout<VMFMA_F32_32x32x16_F16>,
                                                               subgroup_basis = [[1, 1, 4, 1, 1, 1], [0, 1, 2, 4, 5]], promote_operands = [1] }>},

      pv_attrs = {attention_pv_matmul,
                  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
                                                               subgroup_basis = [[1, 1, 4, 1, 1, 1], [0, 1, 2, 3, 5]], promote_operands = [1] }>}
    } -> !transform.any_param

    transform.yield %attention, %config, %decomposition_config : !transform.any_op, !transform.any_param, !transform.any_param
  }

transform.named_sequence @match_attention_f8(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %attention : !transform.any_op

    transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
    %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<?x?x?x?xf8E4M3FNUZ> : !transform.any_value

    %config = transform.param.constant #iree_codegen.compilation_info<
            lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 1, 64, 0, 0, 0], reduction=[0, 0, 0, 0, 0, 64], promote_operands = [1, 2]}>,
            translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                                              workgroup_size = [64, 4]
                                                              subgroup_size = 64 ,
              {llvm_func_attrs = { "amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign" }}>>
    -> !transform.any_param

    %decomposition_config = transform.param.constant {
      qk_attrs = {attention_qk_matmul,
                  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>,
                                                               subgroup_basis = [[1, 1, 4, 1, 1, 1], [0, 1, 2, 4, 5]], promote_operands = [1] }>},
      pv_attrs = {attention_pv_matmul,
                  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.virtual_mma_layout<VMFMA_F32_16x16x32_F8E4M3FNUZ>,
                                                               subgroup_basis = [[1, 1, 4, 1, 1, 1], [0, 1, 2, 3, 5]], promote_operands = [1] }>}
    } -> !transform.any_param

    transform.yield %attention, %config, %decomposition_config : !transform.any_op, !transform.any_param, !transform.any_param
  }

// TUNING_SPEC_BEGIN DO NOT REMOVE

//===----------------------------------------------------------------------===//
// Matmul tuning
//===----------------------------------------------------------------------===//

transform.named_sequence @match_mmt_2048x10240x1280(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  transform.iree.match.has_no_lowering_config %matmul : !transform.any_op

  %batch, %m, %n, %k = transform.iree.match.contraction %matmul,
    lhs_type = i8, rhs_type = i8, output_type = i32,
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>] : !transform.any_op -> !transform.param<i64>
  transform.iree.match.dims_equal %m, [2048] : !transform.param<i64>
  transform.iree.match.dims_equal %n, [10240] : !transform.param<i64>
  transform.iree.match.dims_equal %k, [1280] : !transform.param<i64>
  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                 mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
                                                 subgroup_basis = [[4, 2, 1], [0, 1, 2]],
                                                 reduction = [0, 0, 128],
                                                 workgroup = [128, 320, 0]}>,
    translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
      workgroup_size = [128, 4, 1] subgroup_size = 64,
      {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>
      }>> -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence @match_mmt_2048x1280x5120(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  transform.iree.match.has_no_lowering_config %matmul : !transform.any_op

  %batch, %m, %n, %k = transform.iree.match.contraction %matmul,
    lhs_type = i8, rhs_type = i8, output_type = i32,
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>] : !transform.any_op -> !transform.param<i64>
  transform.iree.match.dims_equal %m, [2048] : !transform.param<i64>
  transform.iree.match.dims_equal %n, [1280] : !transform.param<i64>
  transform.iree.match.dims_equal %k, [5120] : !transform.param<i64>
  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                 mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
                                                 subgroup_basis = [[4, 1, 1], [0, 1, 2]],
                                                 reduction = [0, 0, 256],
                                                 workgroup = [128, 80, 0]}>,
    translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
      workgroup_size = [64, 4, 1] subgroup_size = 64,
      {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>
      }>> -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence @match_mmt_2048x1280x1280(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  transform.iree.match.has_no_lowering_config %matmul : !transform.any_op

  %batch, %m, %n, %k = transform.iree.match.contraction %matmul,
    lhs_type = i8, rhs_type = i8, output_type = i32,
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>] : !transform.any_op -> !transform.param<i64>
  transform.iree.match.dims_equal %m, [2048] : !transform.param<i64>
  transform.iree.match.dims_equal %n, [1280] : !transform.param<i64>
  transform.iree.match.dims_equal %k, [1280] : !transform.param<i64>
  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                 mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
                                                 subgroup_basis = [[2, 2, 1], [0, 1, 2]],
                                                 reduction = [0, 0, 128],
                                                 workgroup = [64, 160, 0]}>,
    translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
      workgroup_size = [256, 1, 1] subgroup_size = 64,
      {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2,
                                                         reorder_workgroups_strategy = <Transpose>>}>
  > -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence @match_mmt_8192x640x640(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  transform.iree.match.has_no_lowering_config %matmul : !transform.any_op

  %batch, %m, %n, %k = transform.iree.match.contraction %matmul,
    lhs_type = i8, rhs_type = i8, output_type = i32,
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>] : !transform.any_op -> !transform.param<i64>
  transform.iree.match.dims_equal %m, [8192] : !transform.param<i64>
  transform.iree.match.dims_equal %n, [640] : !transform.param<i64>
  transform.iree.match.dims_equal %k, [640] : !transform.param<i64>
  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                 mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
                                                 subgroup_basis = [[8, 1, 1], [0, 1, 2]],
                                                 reduction = [0, 0, 64],
                                                 workgroup = [256, 64, 0]}>,
    translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
      workgroup_size = [512, 1, 1] subgroup_size = 64,
      {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>}>
  > -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence @match_mmt_8192x5120x640(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  transform.iree.match.has_no_lowering_config %matmul : !transform.any_op

  %batch, %m, %n, %k = transform.iree.match.contraction %matmul,
    lhs_type = i8, rhs_type = i8, output_type = i32,
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>] : !transform.any_op -> !transform.param<i64>
  transform.iree.match.dims_equal %m, [8192] : !transform.param<i64>
  transform.iree.match.dims_equal %n, [5120] : !transform.param<i64>
  transform.iree.match.dims_equal %k, [640] : !transform.param<i64>
  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                 mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>,
                                                 subgroup_basis = [[2, 4, 1], [0, 1, 2]],
                                                 reduction = [0, 0, 64],
                                                 workgroup = [256, 128, 0]}>,
    translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
      workgroup_size = [512, 1, 1] subgroup_size = 64,
      {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>}>
  > -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

transform.named_sequence @match_mmt_8192x640x2560 (%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  transform.iree.match.has_no_lowering_config %matmul : !transform.any_op

  %batch, %m, %n, %k = transform.iree.match.contraction %matmul,
    lhs_type = i8, rhs_type = i8, output_type = i32,
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>] : !transform.any_op -> !transform.param<i64>
  transform.iree.match.dims_equal %m, [8192] : !transform.param<i64>
  transform.iree.match.dims_equal %n, [640] : !transform.param<i64>
  transform.iree.match.dims_equal %k, [2560] : !transform.param<i64>
  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                 mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
                                                 subgroup_basis = [[8, 1, 1], [0, 1, 2]],
                                                 reduction = [0, 0, 64],
                                                 workgroup = [256, 64, 0]}>,
    translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
      workgroup_size = [512, 1, 1] subgroup_size = 64,
      {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>}>
  > -> !transform.any_param
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

  transform.named_sequence @match_broadcast_rhs_mmt_Bx1024x10240x1280(%generic: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %generic : !transform.any_op

    %batch, %m, %n, %k = transform.iree.match.contraction %generic,
      lhs_type = i8, rhs_type = i8, output_type = i32,
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>] : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %m, [-1, 1024] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [10240] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [1280] : !transform.param<i64>
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                   mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
                                                   subgroup_basis = [[1, 4, 2, 1], [0, 1, 2, 3]],
                                                   reduction = [0, 0, 0, 128],
                                                   workgroup = [1, 128, 320, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
        workgroup_size = [512, 1, 1] subgroup_size = 64,
        {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>}>
    > -> !transform.any_param
    transform.yield %generic, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_broadcast_rhs_mmt_Bx1024x1280x1280(%generic: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %generic : !transform.any_op

    %batch, %m, %n, %k = transform.iree.match.contraction %generic,
      lhs_type = i8, rhs_type = i8, output_type = i32,
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>] : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %m, [-1, 1024] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [1280] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [1280] : !transform.param<i64>
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                   mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
                                                   subgroup_basis = [[1, 2, 2, 1], [0, 1, 2, 3]],
                                                   reduction = [0, 0, 0, 128],
                                                   workgroup = [1, 64, 160, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
        workgroup_size = [256, 1, 1] subgroup_size = 64,
        {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2,
                                                           reorder_workgroups_strategy = <Transpose>>}>
    > -> !transform.any_param
    transform.yield %generic, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_broadcast_rhs_mmt_Bx64x1280x2480(%generic: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %generic : !transform.any_op

    %batch, %m, %n, %k = transform.iree.match.contraction %generic,
      lhs_type = i8, rhs_type = i8, output_type = i32,
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>] : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %m, [-1, 64] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [1280] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [2480] : !transform.param<i64>
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                   mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
                                                   subgroup_basis = [[1, 2, 2, 1], [0, 1, 2, 3]],
                                                   reduction = [0, 0, 0, 128],
                                                   workgroup = [1, 64, 160, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
        workgroup_size = [256, 1, 1] subgroup_size = 64,
        {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2,
                                                           reorder_workgroups_strategy = <Transpose>>
        }>
    > -> !transform.any_param
    transform.yield %generic, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_broadcast_rhs_mmt_Bx4960x640x640(%generic: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %generic : !transform.any_op

    %batch, %m, %n, %k = transform.iree.match.contraction %generic,
      lhs_type = i8, rhs_type = i8, output_type = i32,
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>] : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %m, [-1, 4960] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [640] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [640] : !transform.param<i64>
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                   mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
                                                   subgroup_basis = [[1, 8, 1, 1], [0, 1, 2, 3]],
                                                   reduction = [0, 0, 0, 64],
                                                   workgroup = [1, 256, 64, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
        workgroup_size = [512, 1, 1] subgroup_size = 64,
        {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>}>
    > -> !transform.any_param
    transform.yield %generic, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_broadcast_rhs_mmt_Bx64x640x2480(%generic: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %generic : !transform.any_op

    %batch, %m, %n, %k = transform.iree.match.contraction %generic,
      lhs_type = i8, rhs_type = i8, output_type = i32,
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>] : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %m, [-1, 64] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [640] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [2480] : !transform.param<i64>
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                   mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
                                                   subgroup_basis = [[1, 2, 1, 1], [0, 1, 2, 3]],
                                                   reduction = [0, 0, 0, 128],
                                                   workgroup = [1, 32, 320, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
        workgroup_size = [128, 1, 1] subgroup_size = 64,
        {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>}>
    > -> !transform.any_param
    transform.yield %generic, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_broadcast_rhs_mmt_Bx4096x5120x640(%generic: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %generic : !transform.any_op

    %batch, %m, %n, %k = transform.iree.match.contraction %generic,
      lhs_type = i8, rhs_type = i8, output_type = i32,
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>] : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %m, [-1, 4096] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [5120] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [640] : !transform.param<i64>
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                   mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>,
                                                   subgroup_basis = [[1, 2, 4, 1], [0, 1, 2, 3]],
                                                   reduction = [0, 0, 0, 64],
                                                   workgroup = [1, 256, 128, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
        workgroup_size = [512, 1, 1] subgroup_size = 64,
        {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>}>
    > -> !transform.any_param
    transform.yield %generic, %config : !transform.any_op, !transform.any_param
  }

//===----------------------------------------------------------------------===//
// Contraction tuning
//===----------------------------------------------------------------------===//

  transform.named_sequence @match_matmul_like_Bx20x1024x64x1280_i8xi8xi32(%cont: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %cont : !transform.any_op

    %batch, %m, %n, %k = transform.iree.match.contraction %cont,
    lhs_type = i8, rhs_type = i8, output_type = i32,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                      affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>] : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %m, [-1, 1024] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [20, 64] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [1280] : !transform.param<i64>
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                   mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
                                                   subgroup_basis = [[1, 1, 2, 2, 1], [0, 1, 2, 3, 4]],
                                                   reduction = [0, 0, 0, 0, 128],
                                                   workgroup = [1, 1, 64, 160, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
        workgroup_size = [256, 1, 1] subgroup_size = 64,
        {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2,
                                                           reorder_workgroups_strategy = <Transpose>>
        }>
    > -> !transform.any_param
    transform.yield %cont, %config : !transform.any_op, !transform.any_param
  }


  // Variant of matmul_like_Bx20x1024x64x1280_i8xi8xi32 from Transposed-V.
  transform.named_sequence @match_matmul_like_Bx20x64x1024x1280_i8xi8xi32(%cont: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %cont : !transform.any_op

    %batch, %m, %n, %k = transform.iree.match.contraction %cont,
    lhs_type = i8, rhs_type = i8, output_type = i32,
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                      affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d4)>,
                      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>] : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %m, [-1, 1024] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [20, 64] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [1280] : !transform.param<i64>
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                   mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
                                                   subgroup_basis = [[1, 1, 2, 2, 1], [0, 1, 2, 3, 4]],
                                                   reduction = [0, 0, 0, 0, 128],
                                                   workgroup = [1, 1, 160, 64, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
        workgroup_size = [256, 1, 1] subgroup_size = 64,
        {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2,
                                                           reorder_workgroups_strategy = <Transpose>>
        }>
    > -> !transform.any_param
    transform.yield %cont, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_matmul_like_Bx20x64x64x2048_i8xi8xi32(%cont: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %cont : !transform.any_op

    %batch, %m, %n, %k = transform.iree.match.contraction %cont,
      lhs_type = i8, rhs_type = i8, output_type = i32,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>] : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %m, [-1, 64] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [20, 64] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [2048] : !transform.param<i64>
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                   mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
                                                   subgroup_basis = [[1, 1, 2, 1, 1], [0, 1, 2, 3, 4]],
                                                   reduction = [0, 0, 0, 0, 128],
                                                   workgroup = [1, 1, 32, 320, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
        workgroup_size = [128, 1, 1] subgroup_size = 64,
        {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>}>
    > -> !transform.any_param
    transform.yield %cont, %config : !transform.any_op, !transform.any_param
  }

  // Variant of matmul_like_Bx20x64x64x2048_i8xi8xi32 from Transposed-V.
transform.named_sequence @match_matmul_like_Bx20x64x64x2048_transposev_i8xi8xi32(%cont: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %cont : !transform.any_op

    %batch, %m, %n, %k = transform.iree.match.contraction %cont,
      lhs_type = i8, rhs_type = i8, output_type = i32,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d4)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>] : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %m, [-1, 64] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [20, 64] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [2048] : !transform.param<i64>
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                   mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
                                                   subgroup_basis = [[1, 1, 2, 1, 1], [0, 1, 2, 3, 4]],
                                                   reduction = [0, 0, 0, 0, 128],
                                                   workgroup = [1, 1, 320, 32, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
        workgroup_size = [128, 1, 1] subgroup_size = 64,
        {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>}>
    > -> !transform.any_param
    transform.yield %cont, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_matmul_like_Bx10x4096x64x640_i8xi8xi32(%cont: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %cont : !transform.any_op

    %batch, %m, %n, %k = transform.iree.match.contraction %cont,
      lhs_type = i8, rhs_type = i8, output_type = i32,
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>] : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %m, [-1, 4096] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [10, 64] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [640] : !transform.param<i64>
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                   mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
                                                   subgroup_basis = [[1, 1, 8, 1, 1], [0, 1, 2, 3, 4]],
                                                   reduction = [0, 0, 0, 0, 64],
                                                   workgroup = [1, 1, 256, 64, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
        workgroup_size = [512, 1, 1] subgroup_size = 64,
        {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>}>
    > -> !transform.any_param
    transform.yield %cont, %config : !transform.any_op, !transform.any_param
  }

// TUNING_SPEC_END DO NOT REMOVE

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

  transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %variant_op
        // Attention.
        @match_attention_f16 -> @apply_attn_op_config
        , @match_attention_f8 -> @apply_attn_op_config

        // TUNING_MATCH_BEGIN DO NOT REMOVE

        // Matmul.
        , @match_mmt_2048x10240x1280 -> @apply_op_config
        , @match_mmt_2048x1280x5120 -> @apply_op_config
        , @match_mmt_2048x1280x1280 -> @apply_op_config
        , @match_mmt_8192x640x640 -> @apply_op_config
        , @match_mmt_8192x5120x640 -> @apply_op_config
        //, @match_mmt_8192x640x2560 -> @apply_op_config

        // Convolution.

        // Batch matmul.

        // Broadcast rhs mmt.
        , @match_broadcast_rhs_mmt_Bx4096x5120x640 -> @apply_op_config

        // Carried over from SPX.
        , @match_broadcast_rhs_mmt_Bx1024x10240x1280 -> @apply_op_config
        , @match_broadcast_rhs_mmt_Bx1024x1280x1280 -> @apply_op_config
        , @match_broadcast_rhs_mmt_Bx64x1280x2480 -> @apply_op_config
        , @match_broadcast_rhs_mmt_Bx4960x640x640 -> @apply_op_config
        //, @match_broadcast_rhs_mmt_Bx64x640x2480 -> @apply_op_config


        // Contration.
        , @match_matmul_like_Bx20x1024x64x1280_i8xi8xi32 -> @apply_op_config
        , @match_matmul_like_Bx10x4096x64x640_i8xi8xi32 -> @apply_op_config
        , @match_matmul_like_Bx20x64x64x2048_i8xi8xi32 -> @apply_op_config

        // Transpose-V generated contraction.
        , @match_matmul_like_Bx20x64x1024x1280_i8xi8xi32 -> @apply_op_config
        , @match_matmul_like_Bx20x64x64x2048_transposev_i8xi8xi32 -> @apply_op_config

        // TUNING_MATCH_END DO NOT REMOVE
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} ////  module

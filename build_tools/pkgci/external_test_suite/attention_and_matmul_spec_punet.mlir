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

  transform.named_sequence @match_broadcast_rhs_mmt_i8_i8_i32(
    %root: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
    // transform.print %root {name = "Generic"} : !transform.any_op
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<?x?x?xi8>, %rhs: tensor<?x?xi8>, %out: tensor<?x?x?xi32>):
      %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                                             affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                                             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
                            iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
          ins(%lhs, %rhs : tensor<?x?x?xi8>, tensor<?x?xi8>) outs(%out : tensor<?x?x?xi32>) {
        ^bb0(%in: i8, %in_0: i8, %acc: i32):
          %22 = arith.extsi %in : i8 to i32
          %23 = arith.extsi %in_0 : i8 to i32
          %24 = arith.muli %22, %23 : i32
          %25 = arith.addi %acc, %24 : i32
          linalg.yield %25 : i32
        } -> tensor<?x?x?xi32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %root : !transform.any_op
  }

//===----------------------------------------------------------------------===//
// Attention tuning
//===----------------------------------------------------------------------===//

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

transform.named_sequence @match_attention_f8(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param, !transform.any_param) {
    transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
    %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<?x?x?x?xf8E4M3FNUZ> : !transform.any_value

    %config = transform.param.constant #iree_codegen.compilation_info<
            lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 1, 64, 0, 0, 0], reduction=[0, 0, 0, 0, 64, 0], promote_operands = [1, 2]}>,
            translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                                              workgroup_size = [64, 4]
                                                              subgroup_size = 64 ,
              {llvm_func_attrs = { "amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign" }}>>
    -> !transform.any_param

    %decomposition_config = transform.param.constant {
      qk_attrs = {attention_qk_matmul,
                  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>,
                                                               subgroup_m_count = 4, subgroup_n_count = 1, promote_operands = [1] }>},
      pv_attrs = {attention_pv_matmul,
                  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.virtual_mma_layout<intrinsic = VMFMA_F32_16x16x32_F8E4M3FNUZ>,
                                                               subgroup_m_count = 4, subgroup_n_count = 1, promote_operands = [1] }>}
    } -> !transform.any_param

    transform.yield %attention, %config, %decomposition_config : !transform.any_op, !transform.any_param, !transform.any_param
  }

// TUNING_SPEC_BEGIN DO NOT REMOVE

//===----------------------------------------------------------------------===//
// Matmul tuning
//===----------------------------------------------------------------------===//

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
    %mmt = transform.include @match_broadcast_rhs_mmt_i8_i8_i32 failures(propagate) (%generic) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %generic[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %generic[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<?x1024x1280xi8> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<10240x1280xi8> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                   mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
                                                   subgroup_m_count = 4, subgroup_n_count = 2,
                                                   reduction = [0, 0, 0, 128],
                                                   workgroup = [1, 128, 320, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
        workgroup_size = [512, 1, 1] subgroup_size = 64,
        {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>}>
    > -> !transform.any_param
    transform.yield %generic, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_broadcast_rhs_mmt_Bx1024x1280x1280(%generic: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_broadcast_rhs_mmt_i8_i8_i32 failures(propagate) (%generic) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %generic[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %generic[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<?x1024x1280xi8> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<1280x1280xi8> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                   mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
                                                   subgroup_m_count = 2, subgroup_n_count = 2,
                                                   reduction = [0, 0, 0, 128],
                                                   workgroup = [1, 64, 160, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
        workgroup_size = [256, 1, 1] subgroup_size = 64,
        {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true,
                                                           reorder_workgroups_strategy = <Transpose>>}>
    > -> !transform.any_param
    transform.yield %generic, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_broadcast_rhs_mmt_Bx4096x5120x640(%generic: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_broadcast_rhs_mmt_i8_i8_i32 failures(propagate) (%generic) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %generic[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %generic[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<?x4096x640xi8> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<5120x640xi8> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                   mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>,
                                                   subgroup_m_count = 2, subgroup_n_count = 4,
                                                   reduction = [0, 0, 0, 64],
                                                   workgroup = [1, 256, 128, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
        workgroup_size = [512, 1, 1] subgroup_size = 64,
        {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>}>
    > -> !transform.any_param
    transform.yield %generic, %config : !transform.any_op, !transform.any_param
  }

//===----------------------------------------------------------------------===//
// Contraction tuning
//===----------------------------------------------------------------------===//

  transform.named_sequence @match_matmul_like_Bx20x1024x64x1280_i8xi8xi32(%cont: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %cont {
    ^bb0(%lhs: tensor<?x1024x1280xi8>, %rhs: tensor<20x64x1280xi8>, %out: tensor<?x20x1024x64xi32>):
      %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                                             affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                                             affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
                            iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs : tensor<?x1024x1280xi8>, tensor<20x64x1280xi8>)
        outs(%out : tensor<?x20x1024x64xi32>) {
      ^bb0(%in: i8, %in_0: i8, %acc: i32):
        %18 = arith.extsi %in : i8 to i32
        %19 = arith.extsi %in_0 : i8 to i32
        %20 = arith.muli %18, %19 : i32
        %21 = arith.addi %acc, %20 : i32
        linalg.yield %21 : i32
      } -> tensor<?x20x1024x64xi32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                   mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
                                                   subgroup_m_count = 2, subgroup_n_count = 2,
                                                   reduction = [0, 0, 0, 0, 128],
                                                   workgroup = [1, 1, 64, 160, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
        workgroup_size = [256, 1, 1] subgroup_size = 64,
        {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true,
                                                           reorder_workgroups_strategy = <Transpose>>
        }>
    > -> !transform.any_param
    transform.yield %cont, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_matmul_like_Bx20x64x64x2048_i8xi8xi32(%cont: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %cont {
    ^bb0(%lhs: tensor<?x64x2048xi8>, %rhs: tensor<20x64x2048xi8>, %out: tensor<?x20x64x64xi32>):
      %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                                             affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                                             affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
                            iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs : tensor<?x64x2048xi8>, tensor<20x64x2048xi8>)
        outs(%out : tensor<?x20x64x64xi32>) {
      ^bb0(%in: i8, %in_0: i8, %acc: i32):
        %18 = arith.extsi %in : i8 to i32
        %19 = arith.extsi %in_0 : i8 to i32
        %20 = arith.muli %18, %19 : i32
        %21 = arith.addi %acc, %20 : i32
        linalg.yield %21 : i32
      } -> tensor<?x20x64x64xi32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                   mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
                                                   subgroup_m_count = 2, subgroup_n_count = 1,
                                                   reduction = [0, 0, 0, 0, 128],
                                                   workgroup = [1, 1, 32, 320, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
        workgroup_size = [128, 1, 1] subgroup_size = 64,
        {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>}>
    > -> !transform.any_param
    transform.yield %cont, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_matmul_like_Bx10x4096x64x640_i8xi8xi32(%cont: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %cont {
    ^bb0(%lhs: tensor<?x4096x640xi8>, %rhs: tensor<10x64x640xi8>, %out: tensor<?x10x4096x64xi32>):
      %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                                             affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                                             affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
                            iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs : tensor<?x4096x640xi8>, tensor<10x64x640xi8>)
        outs(%out : tensor<?x10x4096x64xi32>) {
      ^bb0(%in: i8, %in_0: i8, %acc: i32):
        %18 = arith.extsi %in : i8 to i32
        %19 = arith.extsi %in_0 : i8 to i32
        %20 = arith.muli %18, %19 : i32
        %21 = arith.addi %acc, %20 : i32
        linalg.yield %21 : i32
      } -> tensor<?x10x4096x64xi32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                   mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
                                                   subgroup_m_count = 8, subgroup_n_count = 1,
                                                   reduction = [0, 0, 0, 0, 64],
                                                   workgroup = [1, 1, 256, 64, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
        workgroup_size = [512, 1, 1] subgroup_size = 64,
        {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>}>
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

        // Convolution.

        // Batch matmul.

        // Broadcast rhs mmt.
        , @match_broadcast_rhs_mmt_Bx4096x5120x640 -> @apply_op_config

        // Carried over from SPX.
        , @match_broadcast_rhs_mmt_Bx1024x10240x1280 -> @apply_op_config
        , @match_broadcast_rhs_mmt_Bx1024x1280x1280 -> @apply_op_config

        // Contration.
        , @match_matmul_like_Bx20x1024x64x1280_i8xi8xi32 -> @apply_op_config
        , @match_matmul_like_Bx10x4096x64x640_i8xi8xi32 -> @apply_op_config
        , @match_matmul_like_Bx20x64x64x2048_i8xi8xi32 -> @apply_op_config

        // TUNING_MATCH_END DO NOT REMOVE
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} ////  module

module attributes {transform.with_named_sequence} {
//===----------------------------------------------------------------------===//
// Tuning infra
//===----------------------------------------------------------------------===//

  transform.named_sequence @apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield
  }


// TUNING_SPEC_BEGIN DO NOT REMOVE

//===----------------------------------------------------------------------===//
// Matmul tuning
//===----------------------------------------------------------------------===//

  transform.named_sequence @match_contraction_4608x21504x3072_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %arg0 : !transform.any_op

    %batch, %m, %n, %k = transform.iree.match.contraction %arg0,
    lhs_type = bf16, rhs_type = bf16, output_type = f32
    {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>]} : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %m, [4608] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [21504] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [3072] : !transform.param<i64>
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [6, 4, 0], subgroup_basis = [[4, 2, 1], [0, 1, 2]], workgroup = [384, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_contraction_4608x3072x15360_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %arg0 : !transform.any_op

    %batch, %m, %n, %k = transform.iree.match.contraction %arg0,
      lhs_type = bf16, rhs_type = bf16, output_type = f32
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>,
                        affine_map<(d0, d1, d2) -> (d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>]} : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %m, [4608] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [3072] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [15360] : !transform.param<i64>
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [4, 4, 0], subgroup_basis = [[4, 1, 1], [0, 1, 2]], workgroup = [256, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_contraction_4096x12288x3072_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %arg0 : !transform.any_op

    %batch, %m, %n, %k = transform.iree.match.contraction %arg0,
      lhs_type = bf16, rhs_type = bf16, output_type = f32
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                        affine_map<(d0, d1, d2) -> (d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>]} : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %m, [4096] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [12288] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [3072] : !transform.param<i64>
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 6], subgroup = [2, 4, 0], subgroup_basis = [[4, 2, 1], [0, 1, 2]], workgroup = [128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_contraction_4096x3072x12288_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %arg0 : !transform.any_op

    %batch, %m, %n, %k = transform.iree.match.contraction %arg0,
      lhs_type = bf16, rhs_type = bf16, output_type = f32
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                        affine_map<(d0, d1, d2) -> (d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>]} : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %m, [4096] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [3072] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [12288] : !transform.param<i64>
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 6], subgroup = [4, 2, 0], subgroup_basis = [[2, 4, 1], [0, 1, 2]], workgroup = [128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_contraction_72x4096x3072_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %arg0 : !transform.any_op

    %batch, %m, %n, %k = transform.iree.match.contraction %arg0,
      lhs_type = bf16, rhs_type = bf16, output_type = f32
      {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]} : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [72] : !transform.param<i64>
    transform.iree.match.dims_equal %m, [4096] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [128] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [3072] : !transform.param<i64>
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 2], subgroup = [4, 2, 2, 0], subgroup_basis = [[8, 1, 1], [0, 1, 2]], workgroup = [4, 256, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_contraction_4096x3072x3072_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %arg0 : !transform.any_op

    %batch, %m, %n, %k = transform.iree.match.contraction %arg0,
      lhs_type = bf16, rhs_type = bf16, output_type = f32
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                        affine_map<(d0, d1, d2) -> (d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>]} : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %m, [4096] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [3072] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [3072] : !transform.param<i64>
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [4, 6, 0], subgroup_basis = [[2, 2, 1], [0, 1, 2]], workgroup = [128, 192, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_contraction_512x12288x3072_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %arg0 : !transform.any_op

    %batch, %m, %n, %k = transform.iree.match.contraction %arg0,
      lhs_type = bf16, rhs_type = bf16, output_type = f32
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                        affine_map<(d0, d1, d2) -> (d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>]} : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %m, [512] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [12288] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [3072] : !transform.param<i64>
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 6], subgroup = [4, 1, 0], subgroup_basis = [[1, 4, 1], [0, 1, 2]], workgroup = [64, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_contraction_512x3072x12288_bf16xbf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.iree.match.has_no_lowering_config %arg0 : !transform.any_op

    %batch, %m, %n, %k = transform.iree.match.contraction %arg0,
      lhs_type = bf16, rhs_type = bf16, output_type = f32
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                        affine_map<(d0, d1, d2) -> (d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>]} : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %m, [512] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [3072] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [12288] : !transform.param<i64>
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 6], subgroup = [4, 2, 0], subgroup_basis = [[1, 3, 1], [0, 1, 2]], workgroup = [64, 96, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [192, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }

// TUNING_SPEC_END DO NOT REMOVE

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

  transform.named_sequence @__kernel_config(%vairant_op: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %vairant_op

        // TUNING_MATCH_BEGIN DO NOT REMOVE
        @match_contraction_4608x21504x3072_bf16xbf16xf32 -> @apply_op_config // 588
        , @match_contraction_4608x3072x15360_bf16xbf16xf32 -> @apply_op_config // 603
        , @match_contraction_4096x12288x3072_bf16xbf16xf32 -> @apply_op_config // 40
        , @match_contraction_4096x3072x12288_bf16xbf16xf32 -> @apply_op_config // 41
        , @match_contraction_72x4096x3072_bf16xbf16xf32 -> @apply_op_config // 19
        , @match_contraction_4096x3072x3072_bf16xbf16xf32 -> @apply_op_config // 38
        , @match_contraction_512x12288x3072_bf16xbf16xf32 -> @apply_op_config // 44
        , @match_contraction_512x3072x12288_bf16xbf16xf32 -> @apply_op_config // 45

        // TUNING_MATCH_END DO NOT REMOVE

        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// Transform dialect specification for setting tuned conv and matmul configs.

module attributes { transform.with_named_sequence } {

//===----------------------------------------------------------------------===//
// Matmul tuning
//===----------------------------------------------------------------------===//

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
          %8 = arith.extf %in : f16 to f32
          %9 = arith.extf %in_0 : f16 to f32
          %10 = arith.mulf %8, %9 : f32
          %11 = arith.addf %acc, %10 : f32
          linalg.yield %11 : f32
        } -> tensor<?x?xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @match_mmt_f16_f16_f16(%root: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
    // transform.print %root {name = "Generic"} : !transform.any_op
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>, %out: tensor<?x?xf16>):
      %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                            affine_map<(d0, d1, d2) -> (d1, d2)>,
                                            affine_map<(d0, d1, d2) -> (d0, d1)>],
                           iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%lhs, %rhs : tensor<?x?xf16>, tensor<?x?xf16>) outs(%out : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_0: f16, %acc: f16):
          %10 = arith.mulf %in, %in_0 : f16
          %11 = arith.addf %acc, %10 : f16
          linalg.yield %11 : f16
        } -> tensor<?x?xf16>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}, %config: !transform.any_param {transform.readonly}) {
    transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
    // transform.print %op {name = "Applied"} : !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_mmt_2048x10240x1280(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<2048x1280xf16> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<10240x1280xf16> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 320, 32]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [128, 1, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
           intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
           subgroup_m_count = 1, subgroup_n_count = 2>
         , llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_mmt_2048x1280x5120(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<2048x5120xf16> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<1280x5120xf16> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 80, 64]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [64, 2, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
           intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
           subgroup_m_count = 2, subgroup_n_count = 1>
         , llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_mmt_2048x1280x1280(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<2048x1280xf16> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<1280x1280xf16> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 64]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [64, 2, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
           intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
           subgroup_m_count = 2, subgroup_n_count = 1>
         , llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_mmt_8192x5120x640(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<8192x640xf16> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<5120x640xf16> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 32]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [64, 2, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
           intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
           subgroup_m_count = 2, subgroup_n_count = 1>
         }>
      > -> !transform.any_param
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_mmt_8192x640x2560(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<8192x2560xf16> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<640x2560xf16> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 160, 64]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [128, 2, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
           intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
           subgroup_m_count = 2, subgroup_n_count = 2>
         , llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_mmt_8192x640x640(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<8192x640xf16> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<640x640xf16> : !transform.any_value
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 160, 64]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [64, 4, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
           intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
           subgroup_m_count = 4, subgroup_n_count = 1>
         , llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
  }

//===----------------------------------------------------------------------===//
// Convolution tuning
//===----------------------------------------------------------------------===//

  transform.named_sequence @match_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x640(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%lhs: tensor<2x?x?x640xf16>, %rhs: tensor<3x3x640x1280xf16>, %out: tensor<2x32x32x1280xf32>):
      %13 = linalg.conv_2d_nhwc_hwcf { dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64> }
        ins(%lhs, %rhs : tensor<2x?x?x640xf16>, tensor<3x3x640x1280xf16>)
        outs(%out : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 496, 320, 1, 1, 80]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
         workgroup_size = [320, 1, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
              subgroup_m_count = 1, subgroup_n_count = 5>
           , reorder_workgroups = "transpose"}>
      > -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1280(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%lhs: tensor<2x?x?x1280xf16>, %rhs: tensor<3x3x1280x1280xf16>, %out: tensor<2x32x32x1280xf32>):
      %13 = linalg.conv_2d_nhwc_hwcf { dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64> }
        ins(%lhs, %rhs : tensor<2x?x?x1280xf16>, tensor<3x3x1280x1280xf16>)
        outs(%out : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 288, 256, 1, 1, 32]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
         workgroup_size = [256, 1, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
              subgroup_m_count = 1, subgroup_n_count = 4>
           , reorder_workgroups = "transpose", llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1920(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%lhs: tensor<2x?x?x1920xf16>, %rhs: tensor<3x3x1920x1280xf16>, %out: tensor<2x32x32x1280xf32>):
      %13 = linalg.conv_2d_nhwc_hwcf { dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64> }
        ins(%lhs, %rhs : tensor<2x?x?x1920xf16>, tensor<3x3x1920x1280xf16>)
        outs(%out : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 384, 320, 1, 1, 80]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
         workgroup_size = [320, 1, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
              subgroup_m_count = 1, subgroup_n_count = 5>
           , reorder_workgroups = "transpose"}>
      > -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x2560(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%lhs: tensor<2x?x?x2560xf16>, %rhs: tensor<3x3x2560x1280xf16>, %out: tensor<2x32x32x1280xf32>):
      %13 = linalg.conv_2d_nhwc_hwcf { dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64> }
        ins(%lhs, %rhs : tensor<2x?x?x2560xf16>, tensor<3x3x2560x1280xf16>)
        outs(%out : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 512, 320, 1, 1, 80]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
         workgroup_size = [320, 1, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
              subgroup_m_count = 1, subgroup_n_count = 5>
           , reorder_workgroups = "transpose"}>
      > -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_conv_2d_nhwc_hwcf_2x128x128x320x3x3x320(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%lhs: tensor<2x?x?x320xf16>, %rhs: tensor<3x3x320x320xf16>, %out: tensor<2x128x128x320xf32>):
      %13 = linalg.conv_2d_nhwc_hwcf { dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64> }
        ins(%lhs, %rhs : tensor<2x?x?x320xf16>, tensor<3x3x320x320xf16>)
        outs(%out : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 512, 160, 1, 1, 16]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
         workgroup_size = [128, 4, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
              subgroup_m_count = 4, subgroup_n_count = 2>
           , reorder_workgroups = "transpose", llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_conv_2d_nhwc_hwcf_2x64x64x640x3x3x640(%conv: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {
    ^bb0(%lhs: tensor<2x?x?x640xf16>, %rhs: tensor<3x3x640x640xf16>, %out: tensor<2x64x64x640xf32>):
      %13 = linalg.conv_2d_nhwc_hwcf { dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64> }
        ins(%lhs, %rhs : tensor<2x?x?x640xf16>, tensor<3x3x640x640xf16>)
        outs(%out : tensor<2x64x64x640xf32>) -> tensor<2x64x64x640xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 464, 320, 1, 1, 80]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
         workgroup_size = [320, 1, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
              subgroup_m_count = 1, subgroup_n_count = 5>
           , reorder_workgroups = "transpose"}>
      > -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
  }

//===----------------------------------------------------------------------===//
// Batch matmul tuning
//===----------------------------------------------------------------------===//

  transform.named_sequence @match_batch_matmul_64x968x320x640(%batch_matmul: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %batch_matmul {
    ^bb0(%lhs: tensor<64x968x640xf16>, %rhs: tensor<64x640x320xf16>, %out: tensor<64x968x320xf32>):
      %13 = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<64x968x640xf16>, tensor<64x640x320xf16>)
        outs(%out : tensor<64x968x320xf32>) -> tensor<64x968x320xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 128, 64, 64]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute
         workgroup_size = [64, 4, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
              subgroup_m_count = 4, subgroup_n_count = 1>
          , llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %batch_matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_batch_matmul_64x968x640x640(%batch_matmul: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %batch_matmul {
    ^bb0(%lhs: tensor<64x968x640xf16>, %rhs: tensor<64x640x640xf16>, %out: tensor<64x968x640xf32>):
      %13 = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<64x968x640xf16>, tensor<64x640x640xf16>)
        outs(%out : tensor<64x968x640xf32>) -> tensor<64x968x640xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 256, 128, 16]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute
         workgroup_size = [64, 4, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
              subgroup_m_count = 4, subgroup_n_count = 1>
          }>
      > -> !transform.any_param
    transform.yield %batch_matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_batch_matmul_64x968x320x960(%batch_matmul: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %batch_matmul {
    ^bb0(%lhs: tensor<64x968x960xf16>, %rhs: tensor<64x960x320xf16>, %out: tensor<64x968x320xf32>):
      %13 = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<64x968x960xf16>, tensor<64x960x320xf16>)
        outs(%out : tensor<64x968x320xf32>) -> tensor<64x968x320xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 128, 64, 64]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute
         workgroup_size = [64, 4, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
              subgroup_m_count = 4, subgroup_n_count = 1>
           , llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}}>
      > -> !transform.any_param
    transform.yield %batch_matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_batch_matmul_64x242x640x960(%batch_matmul: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %batch_matmul {
    ^bb0(%lhs: tensor<64x242x960xf16>, %rhs: tensor<64x960x640xf16>, %out: tensor<64x242x640xf32>):
      %13 = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<64x242x960xf16>, tensor<64x960x640xf16>)
        outs(%out : tensor<64x242x640xf32>) -> tensor<64x242x640xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 128, 128, 32]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute
         workgroup_size = [128, 2, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
              subgroup_m_count = 2, subgroup_n_count = 2>
           , llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %batch_matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_batch_matmul_64x242x1280x1280(%batch_matmul: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %batch_matmul {
    ^bb0(%lhs: tensor<64x242x1280xf16>, %rhs: tensor<64x1280x1280xf16>, %out: tensor<64x242x1280xf32>):
      %13 = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<64x242x1280xf16>, tensor<64x1280x1280xf16>)
        outs(%out : tensor<64x242x1280xf32>) -> tensor<64x242x1280xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 128, 256, 16]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute
         workgroup_size = [128, 2, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
              subgroup_m_count = 2, subgroup_n_count = 2>
          }>
      > -> !transform.any_param
    transform.yield %batch_matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_batch_matmul_64x242x640x1280(%batch_matmul: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %batch_matmul {
    ^bb0(%lhs: tensor<64x242x1280xf16>, %rhs: tensor<64x1280x640xf16>, %out: tensor<64x242x640xf32>):
      %13 = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<64x242x1280xf16>, tensor<64x1280x640xf16>)
        outs(%out : tensor<64x242x640xf32>) -> tensor<64x242x640xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 128, 128, 32]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute
         workgroup_size = [128, 2, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
              subgroup_m_count = 2, subgroup_n_count = 2>
          }>
      > -> !transform.any_param
    transform.yield %batch_matmul, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_batch_matmul_64x242x640x1920(%batch_matmul: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %batch_matmul {
    ^bb0(%lhs: tensor<64x242x1920xf16>, %rhs: tensor<64x1920x640xf16>, %out: tensor<64x242x640xf32>):
      %13 = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<64x242x1920xf16>, tensor<64x1920x640xf16>)
        outs(%out : tensor<64x242x640xf32>) -> tensor<64x242x640xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
      %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 128, 128, 32]]>,
        translation_info = #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute
         workgroup_size = [128, 2, 1] subgroup_size = 64,
          {mma_schedule = #iree_gpu.mma_schedule<
              intrinsic = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
              subgroup_m_count = 2, subgroup_n_count = 2>
          , llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %batch_matmul, %config : !transform.any_op, !transform.any_param
  }

//===----------------------------------------------------------------------===//
// Contraction tuning
//===----------------------------------------------------------------------===//

  transform.named_sequence @match_contract_3x2x20x1024x64x1280(%contract: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %contract {
    ^bb0(%lhs: tensor<2x1024x1280xf16>, %rhs: tensor<3x20x64x1280xf16>, %out: tensor<3x2x20x1024x64xf32>):
      %20 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d5)>,
                         affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d4, d5)>,
                         affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]
      } ins(%lhs, %rhs : tensor<2x1024x1280xf16>, tensor<3x20x64x1280xf16>)
          outs(%out : tensor<3x2x20x1024x64xf32>) {
      ^bb0(%in: f16, %in_0: f16, %acc: f32):
        %22 = arith.extf %in : f16 to f32
        %23 = arith.extf %in_0 : f16 to f32
        %24 = arith.mulf %22, %23 : f32
        %25 = arith.addf %acc, %24 : f32
        linalg.yield %25 : f32
      } -> tensor<3x2x20x1024x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 256, 384, 32]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [64, 4, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
          intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
          subgroup_m_count = 4, subgroup_n_count = 1>
        , llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %contract, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_contract_3x2x10x4096x64x640(%contract: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %contract {
    ^bb0(%lhs: tensor<2x4096x640xf16>, %rhs: tensor<3x10x64x640xf16>, %out: tensor<3x2x10x4096x64xf32>):
      %20 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d5)>,
                         affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d4, d5)>,
                         affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]
      } ins(%lhs, %rhs : tensor<2x4096x640xf16>, tensor<3x10x64x640xf16>)
          outs(%out : tensor<3x2x10x4096x64xf32>) {
      ^bb0(%in: f16, %in_0: f16, %acc: f32):
        %22 = arith.extf %in : f16 to f32
        %23 = arith.extf %in_0 : f16 to f32
        %24 = arith.mulf %22, %23 : f32
        %25 = arith.addf %acc, %24 : f32
        linalg.yield %25 : f32
      } -> tensor<3x2x10x4096x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 128, 160, 64]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [64, 4, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
          intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
          subgroup_m_count = 4, subgroup_n_count = 1>
        , llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
      > -> !transform.any_param
    transform.yield %contract, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_contract_2x10x64x64x2048(%contract: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %contract {
    ^bb0(%lhs: tensor<2x64x2048xf16>, %rhs: tensor<10x64x2048xf16>, %out: tensor<2x10x64x64xf32>):
        %14 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                           affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                           affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
          iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
        } ins(%lhs, %rhs : tensor<2x64x2048xf16>, tensor<10x64x2048xf16>)
          outs(%out : tensor<2x10x64x64xf32>) {
        ^bb0(%in: f16, %in_0: f16, %acc: f32):
          %16 = arith.extf %in : f16 to f32
          %17 = arith.extf %in_0 : f16 to f32
          %18 = arith.mulf %16, %17 : f32
          %19 = arith.addf %acc, %18 : f32
          linalg.yield %19 : f32
        } -> tensor<2x10x64x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 128, 128, 64]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [128, 2, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
          intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
          subgroup_m_count = 2, subgroup_n_count = 2>
        }>
      > -> !transform.any_param
    transform.yield %contract, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_contract_2x20x64x64x2048(%contract: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %contract {
    ^bb0(%lhs: tensor<2x64x2048xf16>, %rhs: tensor<20x64x2048xf16>, %out: tensor<2x20x64x64xf32>):
        %14 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                           affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                           affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
          iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
        } ins(%lhs, %rhs : tensor<2x64x2048xf16>, tensor<20x64x2048xf16>)
          outs(%out : tensor<2x20x64x64xf32>) {
        ^bb0(%in: f16, %in_0: f16, %acc: f32):
          %16 = arith.extf %in : f16 to f32
          %17 = arith.extf %in_0 : f16 to f32
          %18 = arith.mulf %16, %17 : f32
          %19 = arith.addf %acc, %18 : f32
          linalg.yield %19 : f32
        } -> tensor<2x20x64x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 128, 160, 128]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [128, 2, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
          intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
          subgroup_m_count = 2, subgroup_n_count = 2>
        }>
      > -> !transform.any_param
    transform.yield %contract, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_contract_2x20x1024x64x1280(%contract: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_param) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %contract {
    ^bb0(%lhs: tensor<2x1024x1280xf16>, %rhs: tensor<20x64x1280xf16>, %out: tensor<2x20x1024x64xf32>):
      %20 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
      } ins(%lhs, %rhs : tensor<2x1024x1280xf16>, tensor<20x64x1280xf16>)
          outs(%out : tensor<2x20x1024x64xf32>) {
      ^bb0(%in: f16, %in_0: f16, %acc: f32):
        %22 = arith.extf %in : f16 to f32
        %23 = arith.extf %in_0 : f16 to f32
        %24 = arith.mulf %22, %23 : f32
        %25 = arith.addf %acc, %24 : f32
        linalg.yield %25 : f32
      } -> tensor<2x20x1024x64xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 64, 64]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
        workgroup_size = [64, 4, 1] subgroup_size = 64,
        {mma_schedule = #iree_gpu.mma_schedule<
          intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
          subgroup_m_count = 4, subgroup_n_count = 1>
        }>
      > -> !transform.any_param
    transform.yield %contract, %config : !transform.any_op, !transform.any_param
  }

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

  transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %variant_op
        // Matmul.
        @match_mmt_2048x10240x1280 -> @apply_op_config
        , @match_mmt_2048x1280x5120 -> @apply_op_config
        , @match_mmt_2048x1280x1280 -> @apply_op_config
        , @match_mmt_8192x5120x640 -> @apply_op_config
        , @match_mmt_8192x640x2560 -> @apply_op_config
        , @match_mmt_8192x640x640 -> @apply_op_config

        // Convolution.
        , @match_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x640 -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1280 -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x1920 -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x32x32x1280x3x3x2560 -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x64x64x640x3x3x640 -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x128x128x320x3x3x320 -> @apply_op_config

        // Batch matmul.
        , @match_batch_matmul_64x968x320x640 -> @apply_op_config
        , @match_batch_matmul_64x968x640x640 -> @apply_op_config
        , @match_batch_matmul_64x968x320x960 -> @apply_op_config
        , @match_batch_matmul_64x242x1280x1280 -> @apply_op_config
        , @match_batch_matmul_64x242x640x960 -> @apply_op_config
        , @match_batch_matmul_64x242x640x1280 -> @apply_op_config
        , @match_batch_matmul_64x242x640x1920 -> @apply_op_config

        // Contration.
        , @match_contract_3x2x20x1024x64x1280 -> @apply_op_config
        , @match_contract_3x2x10x4096x64x640 -> @apply_op_config
        , @match_contract_2x10x64x64x2048 -> @apply_op_config
        , @match_contract_2x20x64x64x2048 -> @apply_op_config
        , @match_contract_2x20x1024x64x1280 -> @apply_op_config
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} ////  module

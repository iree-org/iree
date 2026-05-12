// RUN: iree-opt --split-input-file \
// RUN:   --iree-codegen-experimental-verify-pipeline-constraints \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-codegen-insert-smt-constraints,iree-codegen-verify-smt-constraints))' %s \
// RUN:   --verify-diagnostics \
// RUN:   | FileCheck %s

// Test: End-to-end failure from generated constraints.
// This ensures constraints are inserted and that verification reports violations.
// It also catches cases where incorrect knob templates skip verification.
#gpu_target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x4_F32>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target}>
#translation = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute>
    workgroup_size = [64, 1, 1] subgroup_size = 64>

func.func @matmul_e2e_generated_violation(
    %lhs: tensor<128x64xf32>, %rhs: tensor<64x256xf32>)
    -> tensor<128x256xf32>
    attributes {hal.executable.target = #exec_target,
                translation_info = #translation} {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x256xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 0>}
      ins(%cst : f32) outs(%init : tensor<128x256xf32>)
      -> tensor<128x256xf32>
  // expected-error @below {{pipeline constraints violated}}
  // expected-note @below {{dim_0 must be divisible by wg_0 (128 % 30 == 0)}}
  %result = linalg.matmul {
      lowering_config = #iree_gpu.lowering_config<{
          workgroup = [30, 64, 0],
          reduction = [0, 0, 16],
          mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
          subgroup_basis = [[2, 2, 1], [0, 1, 2]]}>,
      root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
      outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
  return %result : tensor<128x256xf32>
}

// -----

#gpu_target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x4_F32>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target}>
#translation = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute>
    workgroup_size = [64, 1, 1] subgroup_size = 64>

func.func @conv_e2e_generated_violation(
    %input: tensor<1x18x18x64xf32>, %filter: tensor<3x3x64x128xf32>)
    -> tensor<1x16x16x128xf32>
    attributes {hal.executable.target = #exec_target,
                translation_info = #translation} {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<1x16x16x128xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 1>}
      ins(%cst : f32) outs(%init : tensor<1x16x16x128xf32>)
      -> tensor<1x16x16x128xf32>
  // expected-error @below {{pipeline constraints violated}}
  // expected-note @below {{dim_2 must be divisible by wg_2 (16 % 30 == 0)}}
  %result = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : tensor<2xi64>,
      lowering_config = #iree_gpu.lowering_config<{
          workgroup = [1, 1, 30, 64, 0, 0, 0],
          reduction = [0, 0, 0, 0, 1, 1, 16],
          mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
          subgroup_basis = [[1, 1, 2, 2, 1, 1, 1],
                            [0, 1, 2, 3, 4, 5, 6]]}>,
      root_op = #iree_codegen.root_op<set = 1>,
      strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x18x18x64xf32>,
                               tensor<3x3x64x128xf32>)
      outs(%fill : tensor<1x16x16x128xf32>) -> tensor<1x16x16x128xf32>
  return %result : tensor<1x16x16x128xf32>
}

// -----

// Test: End-to-end constraint insertion and verification.
// Use the same shape as above but with a divisible workgroup size.
// It should pass verification and have constraints erased.
#gpu_target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x4_F32>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target}>
#translation = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute>
    workgroup_size = [64, 1, 1] subgroup_size = 64>

func.func @matmul_e2e_constraints_erased(
    %lhs: tensor<128x64xf32>, %rhs: tensor<64x256xf32>)
    -> tensor<128x256xf32>
    attributes {hal.executable.target = #exec_target,
                translation_info = #translation} {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x256xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 0>}
      ins(%cst : f32) outs(%init : tensor<128x256xf32>)
      -> tensor<128x256xf32>
  %result = linalg.matmul {
      lowering_config = #iree_gpu.lowering_config<{
          workgroup = [32, 64, 0],
          reduction = [0, 0, 16],
          mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
          subgroup_basis = [[2, 2, 1], [0, 1, 2]]}>,
      root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
      outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
  return %result : tensor<128x256xf32>
}

// CHECK-LABEL: func.func @matmul_e2e_constraints_erased
// CHECK:       linalg.matmul
// CHECK-NOT:   iree_codegen.smt.constraints

func.func @conv_e2e_constraints_erased(
    %input: tensor<1x18x18x64xf32>, %filter: tensor<3x3x64x128xf32>)
    -> tensor<1x16x16x128xf32>
    attributes {hal.executable.target = #exec_target,
                translation_info = #translation} {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<1x16x16x128xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 1>}
      ins(%cst : f32) outs(%init : tensor<1x16x16x128xf32>)
      -> tensor<1x16x16x128xf32>
  %result = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : tensor<2xi64>,
      lowering_config = #iree_gpu.lowering_config<{
          workgroup = [1, 1, 16, 64, 0, 0, 0],
          reduction = [0, 0, 0, 0, 1, 1, 16],
          mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
          subgroup_basis = [[1, 1, 2, 2, 1, 1, 1],
                            [0, 1, 2, 3, 4, 5, 6]]}>,
      root_op = #iree_codegen.root_op<set = 1>,
      strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x18x18x64xf32>,
                               tensor<3x3x64x128xf32>)
      outs(%fill : tensor<1x16x16x128xf32>) -> tensor<1x16x16x128xf32>
  return %result : tensor<1x16x16x128xf32>
}

// CHECK-LABEL: func.func @conv_e2e_constraints_erased
// CHECK:       linalg.conv_2d_nhwc_hwcf
// CHECK-NOT:   iree_codegen.smt.constraints

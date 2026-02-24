// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx950 \
// RUN: --iree-codegen-llvmgpu-use-tile-and-fuse-matmul=true --iree-codegen-llvmgpu-test-tile-and-fuse-vectorize=true \
// RUN: --iree-codegen-llvmgpu-use-igemm=false \
// RUN: --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx950 \
// RUN: --iree-codegen-llvmgpu-use-tile-and-fuse-matmul=true --iree-codegen-llvmgpu-test-tile-and-fuse-vectorize=true \
// RUN: --iree-codegen-llvmgpu-use-igemm=false \
// RUN: --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" \
// RUN: --remarks-filter=".*" %s 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS

#lhs_map = affine_map<(M, N, Ko, Kb) -> (M, Ko, Kb)>
#rhs_map = affine_map<(M, N, Ko, Kb) -> (N, Ko, Kb)>
#scale_m = affine_map<(M, N, Ko, Kb) -> (M, Ko)>
#scale_n = affine_map<(M, N, Ko, Kb) -> (N, Ko)>
#out_map = affine_map<(M, N, Ko, Kb) -> (M, N)>
func.func @scaled_matmul(
    %A: tensor<1024x512x32xf4E2M1FN>, %B: tensor<1024x512x32xf4E2M1FN>, %B_scales: tensor<1024x512xf8E8M0FNU>, %A_scales: tensor<1024x512xf8E8M0FNU>, %C: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
  %0 = linalg.generic {
    indexing_maps = [#lhs_map, #rhs_map, #scale_m, #scale_n, #out_map],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } ins(%A, %B, %A_scales, %B_scales : tensor<1024x512x32xf4E2M1FN>, tensor<1024x512x32xf4E2M1FN>, tensor<1024x512xf8E8M0FNU>, tensor<1024x512xf8E8M0FNU>) outs(%C : tensor<1024x1024xf32>) {
  ^bb0(%a: f4E2M1FN, %b: f4E2M1FN, %a_scale: f8E8M0FNU, %b_scale: f8E8M0FNU, %out: f32):
    %1 = arith.scaling_extf %a, %a_scale : f4E2M1FN, f8E8M0FNU to f32
    %2 = arith.scaling_extf %b, %b_scale : f4E2M1FN, f8E8M0FNU to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<1024x1024xf32>
  return %0 : tensor<1024x1024xf32>
}

// CHECK-LABEL: func.func @scaled_matmul
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>
//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>
//  CHECK-SAME:     promote_operands = [0, 1, 2, 3]
//  CHECK-SAME:     promotion_types = [#iree_gpu.swizzle_operand<copy_config = #iree_gpu.derived_thread_config, swizzle = #iree_codegen.xor_shuffle<256, 32>>, #iree_gpu.swizzle_operand<copy_config = #iree_gpu.derived_thread_config, swizzle = #iree_codegen.xor_shuffle<256, 32>>, #iree_gpu.derived_thread_config, #iree_gpu.derived_thread_config]
//  CHECK-SAME:     reduction = [0, 0, 1, 1]
//  CHECK-SAME:     subgroup = [4, 8, 0, 0]
//  CHECK-SAME:     workgroup = [256, 256, 0, 0]

// CHECK-REMARKS: [Analysis] SharedMemoryUsage
// CHECK-REMARKS-SAME: Category:deduceMMASchedule
// CHECK-REMARKS-SAME: Remark=34816

// -----

#lhs_map = affine_map<(B, M, N, Ko, Kb) -> (B, M, Ko, Kb)>
#rhs_map = affine_map<(B, M, N, Ko, Kb) -> (B, N, Ko, Kb)>
#scale_m = affine_map<(B, M, N, Ko, Kb) -> (B, M, Ko)>
#scale_n = affine_map<(B, M, N, Ko, Kb) -> (B, N, Ko)>
#out_map = affine_map<(B, M, N, Ko, Kb) -> (B, M, N)>
func.func @scaled_matmul_with_batch(
    %A: tensor<4x1024x512x32xf4E2M1FN>, %B: tensor<4x1024x512x32xf4E2M1FN>, %A_scales: tensor<4x1024x512xf8E8M0FNU>, %B_scales: tensor<4x1024x512xf8E8M0FNU>, %C: tensor<4x1024x1024xf32>) -> tensor<4x1024x1024xf32> {
  %0 = linalg.generic {
    indexing_maps = [#lhs_map, #rhs_map, #scale_m, #scale_n, #out_map],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
  } ins(%A, %B, %A_scales, %B_scales : tensor<4x1024x512x32xf4E2M1FN>, tensor<4x1024x512x32xf4E2M1FN>, tensor<4x1024x512xf8E8M0FNU>, tensor<4x1024x512xf8E8M0FNU>) outs(%C : tensor<4x1024x1024xf32>) {
  ^bb0(%a: f4E2M1FN, %b: f4E2M1FN, %a_scale: f8E8M0FNU, %b_scale: f8E8M0FNU, %out: f32):
    %1 = arith.scaling_extf %a, %a_scale : f4E2M1FN, f8E8M0FNU to f32
    %2 = arith.scaling_extf %b, %b_scale : f4E2M1FN, f8E8M0FNU to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<4x1024x1024xf32>
  return %0 : tensor<4x1024x1024xf32>
}

// CHECK-LABEL: func.func @scaled_matmul_with_batch
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>
//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>
//  CHECK-SAME:     promote_operands = [0, 1, 2, 3]
//  CHECK-SAME:     promotion_types = [#iree_gpu.swizzle_operand<copy_config = #iree_gpu.derived_thread_config, swizzle = #iree_codegen.xor_shuffle<256, 32>>, #iree_gpu.swizzle_operand<copy_config = #iree_gpu.derived_thread_config, swizzle = #iree_codegen.xor_shuffle<256, 32>>, #iree_gpu.derived_thread_config, #iree_gpu.derived_thread_config]
//  CHECK-SAME:     reduction = [0, 0, 0, 1, 1]
//  CHECK-SAME:     subgroup = [0, 4, 8, 0, 0]
//  CHECK-SAME:     workgroup = [1, 256, 256, 0, 0]

// CHECK-REMARKS: [Analysis] SharedMemoryUsage
// CHECK-REMARKS-SAME: Category:deduceMMASchedule
// CHECK-REMARKS-SAME: Remark=34816

// -----

#lhs_map = affine_map<(M, N, Ko, Kb) -> (M, Ko, Kb)>
#rhs_map = affine_map<(M, N, Ko, Kb) -> (N, Ko, Kb)>
#scale_m = affine_map<(M, N, Ko, Kb) -> (M, Ko)>
#scale_n = affine_map<(M, N, Ko, Kb) -> (N, Ko)>
#out_map = affine_map<(M, N, Ko, Kb) -> (M, N)>
func.func @scaled_matmul_with_dynamic_red_dim(
    %A: tensor<1024x?x?xf4E2M1FN>, %B: tensor<1024x?x?xf4E2M1FN>, %A_scales: tensor<1024x?xf8E8M0FNU>, %B_scales: tensor<1024x?xf8E8M0FNU>, %C: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
  %0 = linalg.generic {
    indexing_maps = [#lhs_map, #rhs_map, #scale_m, #scale_n, #out_map],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } ins(%A, %B, %A_scales, %B_scales : tensor<1024x?x?xf4E2M1FN>, tensor<1024x?x?xf4E2M1FN>, tensor<1024x?xf8E8M0FNU>, tensor<1024x?xf8E8M0FNU>) outs(%C : tensor<1024x1024xf32>) {
  ^bb0(%a: f4E2M1FN, %b: f4E2M1FN, %a_scale: f8E8M0FNU, %b_scale: f8E8M0FNU, %out: f32):
    %1 = arith.scaling_extf %a, %a_scale : f4E2M1FN, f8E8M0FNU to f32
    %2 = arith.scaling_extf %b, %b_scale : f4E2M1FN, f8E8M0FNU to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<1024x1024xf32>
  return %0 : tensor<1024x1024xf32>
}

// CHECK-LABEL: func.func @scaled_matmul_with_dynamic_red_dim
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64
//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//   CHECK-NOT:     mma_kind

// -----

#lhs_map = affine_map<(b, m, n, ko, kb) -> (b, m, ko, kb)>
#rhs_map = affine_map<(b, m, n, ko, kb) -> (n, ko, kb)>
#scale_lhs = affine_map<(b, m, n, ko, kb) -> (b, m, ko)>
#scale_rhs = affine_map<(b, m, n, ko, kb) -> (n, ko)>
#out_map = affine_map<(b, m, n, ko, kb) -> (b, m, n)>
func.func @scaled_matmul_with_dynamic_batch(
    %A: tensor<?x128x512x32xf4E2M1FN>, %B: tensor<16384x512x32xf4E2M1FN>, %A_scales: tensor<?x128x512xf8E8M0FNU>, %B_scales: tensor<16384x512xf8E8M0FNU>, %C: tensor<?x128x16384xf32>) -> tensor<?x128x16384xf32> {
  %0 = linalg.generic {
    indexing_maps = [#lhs_map, #rhs_map, #scale_lhs, #scale_rhs, #out_map],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
  } ins(%A, %B, %A_scales, %B_scales : tensor<?x128x512x32xf4E2M1FN>, tensor<16384x512x32xf4E2M1FN>, tensor<?x128x512xf8E8M0FNU>, tensor<16384x512xf8E8M0FNU>) outs(%C : tensor<?x128x16384xf32>) {
  ^bb0(%a: f4E2M1FN, %b: f4E2M1FN, %a_scale: f8E8M0FNU, %b_scale: f8E8M0FNU, %out: f32):
    %1 = arith.scaling_extf %a, %a_scale : f4E2M1FN, f8E8M0FNU to f32
    %2 = arith.scaling_extf %b, %b_scale : f4E2M1FN, f8E8M0FNU to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<?x128x16384xf32>
  return %0 : tensor<?x128x16384xf32>
}

// CHECK-LABEL: func.func @scaled_matmul_with_dynamic_batch
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>
//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>
//  CHECK-SAME:     promote_operands = [0, 1, 2, 3]
//  CHECK-SAME:     promotion_types = [#iree_gpu.swizzle_operand<copy_config = #iree_gpu.derived_thread_config, swizzle = #iree_codegen.xor_shuffle<256, 32>>, #iree_gpu.swizzle_operand<copy_config = #iree_gpu.derived_thread_config, swizzle = #iree_codegen.xor_shuffle<256, 32>>, #iree_gpu.derived_thread_config, #iree_gpu.derived_thread_config]
//  CHECK-SAME:     reduction = [0, 0, 0, 1, 1]
//  CHECK-SAME:     subgroup = [0, 4, 4, 0, 0]
//  CHECK-SAME:     workgroup = [1, 128, 256, 0, 0]

// CHECK-REMARKS: [Analysis] SharedMemoryUsage
// CHECK-REMARKS-SAME: Category:deduceMMASchedule
// CHECK-REMARKS-SAME: Remark=26112

// -----

#lhs_map = affine_map<(M, N, Ko, Kb) -> (M, Ko, Kb)>
#rhs_map = affine_map<(M, N, Ko, Kb) -> (N, Ko, Kb)>
#scale_m = affine_map<(M, N, Ko, Kb) -> (M, Ko)>
#scale_n = affine_map<(M, N, Ko, Kb) -> (N, Ko)>
#out_map = affine_map<(M, N, Ko, Kb) -> (M, N)>
func.func @small_scaled_matmul(
    %A: tensor<2x1x1xf4E2M1FN>, %B: tensor<2x1x1xf4E2M1FN>, %A_scales: tensor<2x1xf8E8M0FNU>, %B_scales: tensor<2x1xf8E8M0FNU>, %C: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = linalg.generic {
    indexing_maps = [#lhs_map, #rhs_map, #scale_m, #scale_n, #out_map],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } ins(%A, %B, %A_scales, %B_scales : tensor<2x1x1xf4E2M1FN>, tensor<2x1x1xf4E2M1FN>, tensor<2x1xf8E8M0FNU>, tensor<2x1xf8E8M0FNU>) outs(%C : tensor<2x2xf32>) {
  ^bb0(%a: f4E2M1FN, %b: f4E2M1FN, %a_scale: f8E8M0FNU, %b_scale: f8E8M0FNU, %out: f32):
    %1 = arith.scaling_extf %a, %a_scale : f4E2M1FN, f8E8M0FNU to f32
    %2 = arith.scaling_extf %b, %b_scale : f4E2M1FN, f8E8M0FNU to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @small_scaled_matmul
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>
//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>
//  CHECK-SAME:     promote_operands = [0, 1, 2, 3]
//  CHECK-SAME:     promotion_types = [#iree_gpu.swizzle_operand<copy_config = #iree_gpu.derived_thread_config, swizzle = #iree_codegen.xor_shuffle<256, 32>>, #iree_gpu.swizzle_operand<copy_config = #iree_gpu.derived_thread_config, swizzle = #iree_codegen.xor_shuffle<256, 32>>, #iree_gpu.derived_thread_config, #iree_gpu.derived_thread_config]
//  CHECK-SAME:     reduction = [0, 0, 1, 1]
//  CHECK-SAME:     subgroup = [1, 1, 0, 0]
//  CHECK-SAME:     workgroup = [16, 16, 0, 0]

// CHECK-REMARKS: [Analysis] SharedMemoryUsage
// CHECK-REMARKS-SAME: Category:deduceMMASchedule
// CHECK-REMARKS-SAME: Remark=2176

// -----

module {
  func.func @data_tiled_scaled_mma_inner_tiled(
      %lhs: tensor<1x1x1x2x4x4x16x32xf4E2M1FN>, %rhs: tensor<1x1x1x2x4x4x16x32xf4E2M1FN>,
      %lhs_scales: tensor<1x1x2x4x16x4xf8E8M0FNU>, %rhs_scales: tensor<1x1x2x4x16x4xf8E8M0FNU>,
      %acc: tensor<1x1x2x2x4x16x4xf32>) -> tensor<1x1x2x2x4x16x4xf32> {
    %0 = iree_codegen.inner_tiled ins(%lhs, %rhs, %lhs_scales, %rhs_scales) outs(%acc) {
      indexing_maps = [affine_map<(m, n, k, kb) -> (m, k, kb)>,
                       affine_map<(m, n, k, kb) -> (n, k, kb)>,
                       affine_map<(m, n, k, kb) -> (m, k)>,
                       affine_map<(m, n, k, kb) -> (n, k)>,
                       affine_map<(m, n, k, kb) -> (m, n)>],
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.data_tiled_scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32, subgroups_m = 2, subgroups_n = 2, intrinsics_k = 4, operands_interleaving_intrinsics_k = [2, 3]>,
      semantics = #iree_gpu.mma_semantics<distributed = false, opaque = false>}
      : tensor<1x1x1x2x4x4x16x32xf4E2M1FN>, tensor<1x1x1x2x4x4x16x32xf4E2M1FN>, tensor<1x1x2x4x16x4xf8E8M0FNU>, tensor<1x1x2x4x16x4xf8E8M0FNU> into tensor<1x1x2x2x4x16x4xf32>
      return %0 : tensor<1x1x2x2x4x16x4xf32>
  }
}

// CHECK-LABEL: func.func @data_tiled_scaled_mma_inner_tiled
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64
//  CHECK-SAME:   {gpu_pipeline_options = #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}
//       CHECK:   iree_codegen.inner_tiled {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 1, 1]
//  CHECK-SAME:     workgroup = [1, 1, 0, 0]

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>
func.func @data_tiled_scaled_mma_inner_tiled_with_copy(
    %lhs: tensor<1x1x1x2x4x4x16x32xf4E2M1FN>, %rhs: tensor<1x1x1x2x4x4x16x32xf4E2M1FN>,
    %lhs_scales: tensor<1x1x2x4x16x4xf8E8M0FNU>, %rhs_scales: tensor<1x1x2x4x16x4xf8E8M0FNU>,
    %acc: tensor<1x1x2x2x4x16x4xf32>) -> tensor<1x1x2x2x4x16x4xf16> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs, %lhs_scales, %rhs_scales) outs(%acc) {
    indexing_maps = [affine_map<(m, n, k, kb) -> (m, k, kb)>,
                     affine_map<(m, n, k, kb) -> (n, k, kb)>,
                     affine_map<(m, n, k, kb) -> (m, k)>,
                     affine_map<(m, n, k, kb) -> (n, k)>,
                     affine_map<(m, n, k, kb) -> (m, n)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.data_tiled_scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32, subgroups_m = 2, subgroups_n = 2, intrinsics_k = 4, operands_interleaving_intrinsics_k = [2, 3]>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = false>}
    : tensor<1x1x1x2x4x4x16x32xf4E2M1FN>, tensor<1x1x1x2x4x4x16x32xf4E2M1FN>, tensor<1x1x2x4x16x4xf8E8M0FNU>, tensor<1x1x2x4x16x4xf8E8M0FNU> into tensor<1x1x2x2x4x16x4xf32>
  %empty = tensor.empty() : tensor<1x1x2x2x4x16x4xf16>
  %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%0 : tensor<1x1x2x2x4x16x4xf32>) outs(%empty : tensor<1x1x2x2x4x16x4xf16>) {
  ^bb0(%in: f32, %out: f16):
    %trunc = arith.truncf %in : f32 to f16
    linalg.yield %trunc : f16
  } -> tensor<1x1x2x2x4x16x4xf16>
  %copy_out = tensor.empty() : tensor<1x1x2x2x4x16x4xf16>
  %2 = linalg.copy ins(%1 : tensor<1x1x2x2x4x16x4xf16>) outs(%copy_out : tensor<1x1x2x2x4x16x4xf16>) -> tensor<1x1x2x2x4x16x4xf16>
  return %2 : tensor<1x1x2x2x4x16x4xf16>
}

// CHECK-LABEL: func.func @data_tiled_scaled_mma_inner_tiled_with_copy
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
//       CHECK:   iree_codegen.inner_tiled {{.*}}lowering_config = #iree_gpu.lowering_config
//       CHECK:   linalg.copy
//   CHECK-NOT:   lowering_config

// -----

// Test that scaled matmul with an existing accumulator (matmul_accumulate)
// gets smaller tiles than a zero-initialized matmul to account for accumulator
// memory in workgroup memory (LDS). Without this, 256x256 tiles would be
// selected, but the 256x256 f32 accumulator (256KB) exceeds gfx950's 160KB
// workgroup memory limit (max_workgroup_memory_bytes).
#lhs_map_acc = affine_map<(M, N, Ko, Kb) -> (M, Ko, Kb)>
#rhs_map_acc = affine_map<(M, N, Ko, Kb) -> (N, Ko, Kb)>
#scale_m_acc = affine_map<(M, N, Ko, Kb) -> (M, Ko)>
#scale_n_acc = affine_map<(M, N, Ko, Kb) -> (N, Ko)>
#out_map_acc = affine_map<(M, N, Ko, Kb) -> (M, N)>
func.func @scaled_matmul_accumulate(
    %C_dispatch: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1024x1024xf32>>,
    %A: tensor<1024x512x32xf4E2M1FN>, %B: tensor<1024x512x32xf4E2M1FN>,
    %A_scales: tensor<1024x512xf8E8M0FNU>, %B_scales: tensor<1024x512xf8E8M0FNU>) {
  // Load accumulator from a readwrite buffer - this triggers hasExistingAccumulator
  %C = iree_tensor_ext.dispatch.tensor.load %C_dispatch, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1024x1024xf32>> -> tensor<1024x1024xf32>
  %result = linalg.generic {
    indexing_maps = [#lhs_map_acc, #rhs_map_acc, #scale_m_acc, #scale_n_acc, #out_map_acc],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } ins(%A, %B, %A_scales, %B_scales : tensor<1024x512x32xf4E2M1FN>, tensor<1024x512x32xf4E2M1FN>, tensor<1024x512xf8E8M0FNU>, tensor<1024x512xf8E8M0FNU>) outs(%C : tensor<1024x1024xf32>) {
  ^bb0(%a: f4E2M1FN, %b: f4E2M1FN, %a_scale: f8E8M0FNU, %b_scale: f8E8M0FNU, %out: f32):
    %1 = arith.scaling_extf %a, %a_scale : f4E2M1FN, f8E8M0FNU to f32
    %2 = arith.scaling_extf %b, %b_scale : f4E2M1FN, f8E8M0FNU to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<1024x1024xf32>
  // Store result back to the same buffer
  iree_tensor_ext.dispatch.tensor.store %result, %C_dispatch, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1]
      : tensor<1024x1024xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1024x1024xf32>>
  return
}

// CHECK-LABEL: func.func @scaled_matmul_accumulate
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64
//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>
//  CHECK-SAME:     promote_operands = [0, 1, 2, 3]
//  CHECK-SAME:     promotion_types = [#iree_gpu.swizzle_operand<copy_config = #iree_gpu.derived_thread_config, swizzle = #iree_codegen.xor_shuffle<256, 32>>, #iree_gpu.swizzle_operand<copy_config = #iree_gpu.derived_thread_config, swizzle = #iree_codegen.xor_shuffle<256, 32>>, #iree_gpu.derived_thread_config, #iree_gpu.derived_thread_config]
//  CHECK-SAME:     reduction = [0, 0, 1, 1]
//  CHECK-SAME:     subgroup = [2, 8, 0, 0]
//       CHECK:     workgroup = [128, 256, 0, 0]

// CHECK-REMARKS: [Analysis] SharedMemoryUsage
// CHECK-REMARKS-SAME: Category:deduceMMASchedule
// CHECK-REMARKS-SAME: Remark=157184

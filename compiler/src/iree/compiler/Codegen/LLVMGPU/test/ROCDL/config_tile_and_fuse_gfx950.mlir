// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx950 \
// RUN: --iree-codegen-llvmgpu-early-tile-and-fuse-matmul=true --iree-codegen-llvmgpu-test-tile-and-fuse-vectorize=true \
// RUN: --iree-codegen-llvmgpu-use-igemm=false \
// RUN: --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

#lhs_map = affine_map<(M, N, Ko, Kb) -> (M, Ko, Kb)>
#rhs_map = affine_map<(M, N, Ko, Kb) -> (N, Ko, Kb)>
#scale_m = affine_map<(M, N, Ko, Kb) -> (M, Ko)>
#scale_n = affine_map<(M, N, Ko, Kb) -> (N, Ko)>
#out_map = affine_map<(M, N, Ko, Kb) -> (M, N)>
func.func @scaled_matmul(
    %A: tensor<1024x32x32xf4E2M1FN>, %B: tensor<1024x32x32xf4E2M1FN>, %A_scales: tensor<1024x32xf8E8M0FNU>, %B_scales: tensor<1024x32xf8E8M0FNU>, %C: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
  %0 = linalg.generic {
    indexing_maps = [#lhs_map, #rhs_map, #scale_m, #scale_n, #out_map],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } ins(%A, %B, %A_scales, %B_scales : tensor<1024x32x32xf4E2M1FN>, tensor<1024x32x32xf4E2M1FN>, tensor<1024x32xf8E8M0FNU>, tensor<1024x32xf8E8M0FNU>) outs(%C : tensor<1024x1024xf32>) {
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
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>
//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>
//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 8, 1]
//  CHECK-SAME:     subgroup = [4, 4, 0, 0]
//  CHECK-SAME:     workgroup = [128, 128, 0, 0]

// -----

#lhs_map = affine_map<(B, M, N, Ko, Kb) -> (B, M, Ko, Kb)>
#rhs_map = affine_map<(B, M, N, Ko, Kb) -> (B, N, Ko, Kb)>
#scale_m = affine_map<(B, M, N, Ko, Kb) -> (B, M, Ko)>
#scale_n = affine_map<(B, M, N, Ko, Kb) -> (B, N, Ko)>
#out_map = affine_map<(B, M, N, Ko, Kb) -> (B, M, N)>
func.func @scaled_matmul_with_batch(
    %A: tensor<4x1024x32x32xf4E2M1FN>, %B: tensor<4x1024x32x32xf4E2M1FN>, %A_scales: tensor<4x1024x32xf8E8M0FNU>, %B_scales: tensor<4x1024x32xf8E8M0FNU>, %C: tensor<4x1024x1024xf32>) -> tensor<4x1024x1024xf32> {
  %0 = linalg.generic {
    indexing_maps = [#lhs_map, #rhs_map, #scale_m, #scale_n, #out_map],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
  } ins(%A, %B, %A_scales, %B_scales : tensor<4x1024x32x32xf4E2M1FN>, tensor<4x1024x32x32xf4E2M1FN>, tensor<4x1024x32xf8E8M0FNU>, tensor<4x1024x32xf8E8M0FNU>) outs(%C : tensor<4x1024x1024xf32>) {
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
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>
//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>
//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 0, 8, 1]
//  CHECK-SAME:     subgroup = [0, 4, 4, 0, 0]
//  CHECK-SAME:     workgroup = [1, 128, 128, 0, 0]

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
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>
//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>
//  CHECK-SAME:     promote_operands = [0, 1, 2]
//  CHECK-SAME:     reduction = [0, 0, 1, 1]
//  CHECK-SAME:     subgroup = [1, 1, 0, 0]
//  CHECK-SAME:     workgroup = [16, 16, 0, 0]

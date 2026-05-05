// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx950 \
// RUN: --iree-codegen-llvmgpu-use-tile-and-fuse-matmul=true --iree-codegen-llvmgpu-test-tile-and-fuse-vectorize=true \
// RUN: --iree-codegen-llvmgpu-use-igemm=false \
// RUN: --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx950 \
// RUN: --iree-codegen-llvmgpu-use-tile-and-fuse-matmul=true --iree-codegen-llvmgpu-test-tile-and-fuse-vectorize=true \
// RUN: --iree-codegen-llvmgpu-use-igemm=false \
// RUN: --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" \
// RUN: --remarks-filter=".*" %s 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS

// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx950 \
// RUN: --iree-codegen-llvmgpu-use-tile-and-fuse-matmul=true --iree-codegen-llvmgpu-test-tile-and-fuse-vectorize=true \
// RUN: --iree-codegen-llvmgpu-use-igemm=false --iree-llvmgpu-use-direct-load=true \
// RUN: --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s \
// RUN: | FileCheck %s --check-prefix=CHECK-DIRECT-LOAD

// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx950 \
// RUN: --iree-codegen-llvmgpu-use-tile-and-fuse-matmul=true --iree-codegen-llvmgpu-test-tile-and-fuse-vectorize=true \
// RUN: --iree-codegen-llvmgpu-use-igemm=false --iree-llvmgpu-use-direct-load=true --iree-llvmgpu-prefetch-num-stages=2 \
// RUN: --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" \
// RUN: --remarks-filter=".*" %s 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS-DIRECT-LOAD-2

// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx950 \
// RUN: --iree-codegen-llvmgpu-use-tile-and-fuse-matmul=true --iree-codegen-llvmgpu-test-tile-and-fuse-vectorize=true \
// RUN: --iree-codegen-llvmgpu-use-igemm=false --iree-llvmgpu-use-direct-load=true --iree-llvmgpu-prefetch-num-stages=3 \
// RUN: --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" \
// RUN: --remarks-filter=".*" %s 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS-DIRECT-LOAD-3

// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=mi355x@hip \
// RUN: --iree-codegen-llvmgpu-use-tile-and-fuse-matmul=true --iree-codegen-llvmgpu-test-tile-and-fuse-vectorize=true \
// RUN: --iree-codegen-llvmgpu-use-igemm=false \
// RUN: --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s --check-prefix=MI355X

// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx950 \
// RUN: --iree-codegen-llvmgpu-use-igemm=true --iree-codegen-llvmgpu-test-tile-and-fuse-vectorize=true \
// RUN: --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s --check-prefix=IGEMM

// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx950 \
// RUN: --iree-codegen-llvmgpu-use-igemm=true --iree-codegen-llvmgpu-test-tile-and-fuse-vectorize=true \
// RUN: --iree-llvmgpu-use-direct-load=true \
// RUN: --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s \
// RUN: | FileCheck %s --check-prefix=IGEMM-DIRECT-LOAD

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
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [512, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>
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

// CHECK-REMARKS-DIRECT-LOAD-2: [Analysis] SharedMemoryUsage
// CHECK-REMARKS-DIRECT-LOAD-2-SAME: Category:deduceMMASchedule
// CHECK-REMARKS-DIRECT-LOAD-2-SAME: Remark=34816

// CHECK-REMARKS-DIRECT-LOAD-3: [Analysis] SharedMemoryUsage
// CHECK-REMARKS-DIRECT-LOAD-3-SAME: Category:deduceMMASchedule
// CHECK-REMARKS-DIRECT-LOAD-3-SAME: Remark=34816

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
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [512, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>
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

// CHECK-REMARKS-DIRECT-LOAD-2: [Analysis] SharedMemoryUsage
// CHECK-REMARKS-DIRECT-LOAD-2-SAME: Category:deduceMMASchedule
// CHECK-REMARKS-DIRECT-LOAD-2-SAME: Remark=34816

// CHECK-REMARKS-DIRECT-LOAD-3: [Analysis] SharedMemoryUsage
// CHECK-REMARKS-DIRECT-LOAD-3-SAME: Category:deduceMMASchedule
// CHECK-REMARKS-DIRECT-LOAD-3-SAME: Remark=34816

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
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [64, 1, 1] subgroup_size = 64
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
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [512, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>
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

// CHECK-REMARKS-DIRECT-LOAD-2: [Analysis] SharedMemoryUsage
// CHECK-REMARKS-DIRECT-LOAD-2-SAME: Category:deduceMMASchedule
// CHECK-REMARKS-DIRECT-LOAD-2-SAME: Remark=26112

// CHECK-REMARKS-DIRECT-LOAD-3: [Analysis] SharedMemoryUsage
// CHECK-REMARKS-DIRECT-LOAD-3-SAME: Category:deduceMMASchedule
// CHECK-REMARKS-DIRECT-LOAD-3-SAME: Remark=26112

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
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [64, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>
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

// CHECK-REMARKS-DIRECT-LOAD-2: [Analysis] SharedMemoryUsage
// CHECK-REMARKS-DIRECT-LOAD-2-SAME: Category:deduceMMASchedule
// CHECK-REMARKS-DIRECT-LOAD-2-SAME: Remark=2176

// CHECK-REMARKS-DIRECT-LOAD-3: [Analysis] SharedMemoryUsage
// CHECK-REMARKS-DIRECT-LOAD-3-SAME: Category:deduceMMASchedule
// CHECK-REMARKS-DIRECT-LOAD-3-SAME: Remark=2176

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
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [256, 1, 1] subgroup_size = 64
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
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse>
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
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [512, 1, 1] subgroup_size = 64
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

// CHECK-REMARKS-DIRECT-LOAD-2: [Analysis] SharedMemoryUsage
// CHECK-REMARKS-DIRECT-LOAD-2-SAME: Category:deduceMMASchedule
// CHECK-REMARKS-DIRECT-LOAD-2-SAME: Remark=157184

// CHECK-REMARKS-DIRECT-LOAD-3: [Analysis] SharedMemoryUsage
// CHECK-REMARKS-DIRECT-LOAD-3-SAME: Category:deduceMMASchedule
// CHECK-REMARKS-DIRECT-LOAD-3-SAME: Remark=157184

// -----

// Very large f16 matmul — compute-bound, so picks 32x32x16 (higher compute per
// instruction, lower VGPR pressure than 16x16x32).
func.func @matmul_f16_compute_bound(
    %arg0: tensor<16384x16384xf16>,
    %arg1: tensor<16384x16384xf16>,
    %arg2: tensor<16384x16384xf32>) -> tensor<16384x16384xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<16384x16384xf16>, tensor<16384x16384xf16>)
                      outs(%arg2 : tensor<16384x16384xf32>) -> tensor<16384x16384xf32>
  return %0 : tensor<16384x16384xf32>
}
// CHECK-LABEL: func.func @matmul_f16_compute_bound
// CHECK-SAME:   #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse>
// CHECK:   lowering_config = #iree_gpu.lowering_config
// CHECK-SAME: mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x16_F16>

// CHECK-REMARKS: [Analysis] SharedMemoryUsage
// CHECK-REMARKS-SAME: Category:deduceMMASchedule
// CHECK-REMARKS-SAME: Remark=32768

// CHECK-REMARKS-DIRECT-LOAD-2: [Analysis] SharedMemoryUsage
// CHECK-REMARKS-DIRECT-LOAD-2-SAME: Category:deduceMMASchedule
// CHECK-REMARKS-DIRECT-LOAD-2-SAME: Remark=65536

// CHECK-REMARKS-DIRECT-LOAD-3: [Analysis] SharedMemoryUsage
// CHECK-REMARKS-DIRECT-LOAD-3-SAME: Category:deduceMMASchedule
// CHECK-REMARKS-DIRECT-LOAD-3-SAME: Remark=98304

// -----

// MI355X-specific (CDNA4) heuristic tests: MI355X targets gfx950 with chip
// info (wgpCount=256), enabling utilization-aware MNT boosting for balanced
// large GEMMs. These tests verify that the boosted config differs from bare
// gfx950 (which has no chip info).

// LargeGemm — symmetric (4096x4096x4096)
// Balanced K (K == M == N), so MNT gets boosted to 32.
func.func @matmul_large_symmetric_f16(
    %arg0: tensor<4096x4096xf16>,
    %arg1: tensor<4096x4096xf16>) -> tensor<4096x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<4096x4096xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  %result = linalg.matmul ins(%arg0, %arg1 : tensor<4096x4096xf16>, tensor<4096x4096xf16>)
                          outs(%fill : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  return %result : tensor<4096x4096xf32>
}

// MI355X-LABEL: func.func @matmul_large_symmetric_f16
//  MI355X-SAME:   #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse>
//  MI355X-SAME:   workgroup_size = [256, 1, 1] subgroup_size = 64
//       MI355X:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  MI355X-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>
//  MI355X-SAME:     promote_operands = [0, 1]
//  MI355X-SAME:     reduction = [0, 0, 1]
//  MI355X-SAME:     subgroup = [4, 8, 0]
//  MI355X-SAME:     workgroup = [128, 256, 0]

// -----

// LargeGemm — tall-M (21760x3840x3840)
// Balanced K (K == N < M), MNT boost applies.
func.func @matmul_large_tall_m_f16(
    %arg0: tensor<21760x3840xf16>,
    %arg1: tensor<3840x3840xf16>) -> tensor<21760x3840xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<21760x3840xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<21760x3840xf32>) -> tensor<21760x3840xf32>
  %result = linalg.matmul ins(%arg0, %arg1 : tensor<21760x3840xf16>, tensor<3840x3840xf16>)
                          outs(%fill : tensor<21760x3840xf32>) -> tensor<21760x3840xf32>
  return %result : tensor<21760x3840xf32>
}

// MI355X-LABEL: func.func @matmul_large_tall_m_f16
//  MI355X-SAME:   #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse>
//  MI355X-SAME:   workgroup_size = [256, 1, 1] subgroup_size = 64
//       MI355X:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  MI355X-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>
//  MI355X-SAME:     promote_operands = [0, 1]
//  MI355X-SAME:     reduction = [0, 0, 1]
//  MI355X-SAME:     subgroup = [4, 8, 0]
//  MI355X-SAME:     workgroup = [128, 256, 0]

// -----

// LargeGemm — wide-N (4096x8192x2048)
// Balanced K (K < max(M, N)), MNT boost applies.
func.func @matmul_large_wide_n_f16(
    %arg0: tensor<4096x2048xf16>,
    %arg1: tensor<2048x8192xf16>) -> tensor<4096x8192xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<4096x8192xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4096x8192xf32>) -> tensor<4096x8192xf32>
  %result = linalg.matmul ins(%arg0, %arg1 : tensor<4096x2048xf16>, tensor<2048x8192xf16>)
                          outs(%fill : tensor<4096x8192xf32>) -> tensor<4096x8192xf32>
  return %result : tensor<4096x8192xf32>
}

// MI355X-LABEL: func.func @matmul_large_wide_n_f16
//  MI355X-SAME:   #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse>
//  MI355X-SAME:   workgroup_size = [256, 1, 1] subgroup_size = 64
//       MI355X:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  MI355X-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>
//  MI355X-SAME:     promote_operands = [0, 1]
//  MI355X-SAME:     reduction = [0, 0, 1]
//  MI355X-SAME:     subgroup = [4, 8, 0]
//  MI355X-SAME:     workgroup = [128, 256, 0]

// -----

// LargeGemm — very tall-M with large K (150000x4096x16384)
// K > max(M, N) so K-dominated — MNT boost does NOT apply.
// Requires padding since 150000 is not a multiple of tile size.
func.func @matmul_large_very_tall_m_f16(
    %arg0: tensor<150000x16384xf16>,
    %arg1: tensor<16384x4096xf16>) -> tensor<150000x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<150000x4096xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<150000x4096xf32>) -> tensor<150000x4096xf32>
  %result = linalg.matmul ins(%arg0, %arg1 : tensor<150000x16384xf16>, tensor<16384x4096xf16>)
                          outs(%fill : tensor<150000x4096xf32>) -> tensor<150000x4096xf32>
  return %result : tensor<150000x4096xf32>
}

// MI355X-LABEL: func.func @matmul_large_very_tall_m_f16
//  MI355X-SAME:   #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse>
//  MI355X-SAME:   workgroup_size = [256, 1, 1] subgroup_size = 64
//       MI355X:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  MI355X-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>
//  MI355X-SAME:     padding = [128, 256, 32]
//  MI355X-SAME:     promote_operands = [0, 1]
//  MI355X-SAME:     reduction = [0, 0, 1]
//  MI355X-SAME:     subgroup = [4, 8, 0]
//  MI355X-SAME:     workgroup = [128, 256, 0]

// -----

// Small M*N matmul: M=8, N=8, K=5000. Both M and N need padding.
// Without the M*N utilization rule, this picks MFMA_F32_32x32x8_F16
// (6.25% util). With it, picks MFMA_F32_16x16x16_F16 (25% util, 4x better).
func.func @small_mn_matmul(%lhs: tensor<8x5000xf16>, %rhs: tensor<5000x8xf16>, %out: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %result = linalg.matmul ins(%lhs, %rhs : tensor<8x5000xf16>, tensor<5000x8xf16>) outs(%out : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}
// CHECK-LABEL: func.func @small_mn_matmul
// CHECK:         linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
// CHECK-SAME:      mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
// CHECK-SAME:      padding = [16, 16, 16]
// CHECK-SAME:      subgroup = [1, 1, 0]
// CHECK-SAME:      workgroup = [16, 16, 0]

// -----

// Small-channel grouped convolution (weight backward): 32 groups, 8 in/out channels per group.
// M product (8) <= 2*kVerySkinnyDimThreshold so block intrinsics are allowed and preferred.
// Picks MFMA_F32_4x4x4x16B_F16 (block intrinsic) for small M/N channels.
#map_gc = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d2 + d6, d3 + d7, d0, d4)>
#map_gc1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d7, d0, d1)>
#map_gc2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
func.func @group_conv_small_channels(%arg0: tensor<32x102x102x32x8xf16>, %arg1: tensor<32x100x100x32x8xf16>, %arg2: tensor<32x8x3x3x8xf32>) -> tensor<32x8x3x3x8xf32> {
  %0 = linalg.generic {indexing_maps = [#map_gc, #map_gc1, #map_gc2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<32x102x102x32x8xf16>, tensor<32x100x100x32x8xf16>) outs(%arg2 : tensor<32x8x3x3x8xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %1 = arith.extf %in : f16 to f32
    %2 = arith.extf %in_0 : f16 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<32x8x3x3x8xf32>
  return %0 : tensor<32x8x3x3x8xf32>
}
// IGEMM-LABEL: func.func @group_conv_small_channels
// IGEMM:         linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
// IGEMM-SAME:      mma_kind = #iree_gpu.mma_layout<MFMA_F32_4x4x4x16B_F16>
// IGEMM-SAME:      promote_operands = [0, 1]
// IGEMM-SAME:      reduction = [0, 0, 0, 0, 0, 32]
// IGEMM-SAME:      subgroup = [1, 1, 1, 1, 1, 0]
// IGEMM-SAME:      workgroup = [16, 8, 1, 1, 8, 0]

// -----

// Grouped convolution with 10 channels per group: M product (10) > 2*kVerySkinnyDimThreshold
// so block intrinsics are skipped. Falls back to MFMA_F32_16x16x16_F16 with a larger
// batch tile (workgroup[0]=4) to distribute the batch=32 dimension across workgroups.
#map_gc_10ch = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d2 + d6, d3 + d7, d0, d4)>
#map_gc1_10ch = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d7, d0, d1)>
#map_gc2_10ch = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
func.func @group_conv_10_channels(%arg0: tensor<32x102x102x32x10xf16>, %arg1: tensor<32x100x100x32x10xf16>, %arg2: tensor<32x10x3x3x10xf32>) -> tensor<32x10x3x3x10xf32> {
  %0 = linalg.generic {indexing_maps = [#map_gc_10ch, #map_gc1_10ch, #map_gc2_10ch], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<32x102x102x32x10xf16>, tensor<32x100x100x32x10xf16>) outs(%arg2 : tensor<32x10x3x3x10xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %1 = arith.extf %in : f16 to f32
    %2 = arith.extf %in_0 : f16 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<32x10x3x3x10xf32>
  return %0 : tensor<32x10x3x3x10xf32>
}
// IGEMM-LABEL: func.func @group_conv_10_channels
// IGEMM:         linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
// IGEMM-SAME:      mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
// IGEMM-SAME:      padding = [4, 16, 1, 3, 16, 128]
// IGEMM-SAME:      promote_operands = [0, 1]
// IGEMM-SAME:      reduction = [0, 0, 0, 0, 0, 8]
// IGEMM-SAME:      subgroup = [0, 1, 1, 1, 1, 0]
// IGEMM-SAME:      workgroup = [4, 16, 1, 3, 16, 0]

// -----

// BF16 matmul with DMA. Both LHS and RHS are not transposed, so only LHS gets XOR swizzle.
func.func @matmul_bf16(
    %arg0: tensor<4096x4096xbf16>,
    %arg1: tensor<4096x4096xbf16>,
    %arg2: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<4096x4096xbf16>, tensor<4096x4096xbf16>)
                      outs(%arg2 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  return %0 : tensor<4096x4096xf32>
}

// CHECK-DIRECT-LOAD-LABEL: func.func @matmul_bf16
// CHECK-DIRECT-LOAD:       linalg.matmul {lowering_config = #iree_gpu.lowering_config
// CHECK-DIRECT-LOAD-SAME:    promotion_types = [#iree_gpu.swizzle_operand<copy_config = #iree_gpu.use_global_load_dma, swizzle = #iree_codegen.xor_shuffle<64, 8>>, #iree_gpu.use_global_load_dma]

// -----

// BF16 1x1 conv with DMA. The MMA intrinsic (MFMA_F32_32x32x8_BF16) is not in
// the tuned swizzle table, so no XOR swizzle should be applied -- only plain
// use_global_load_dma.
func.func @conv_bf16_no_untuned_swizzle(
    %arg0: tensor<16x96x64x40xbf16>,
    %arg1: tensor<40x1x1x40xbf16>) -> tensor<16x96x64x40xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<16x96x64x40xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<16x96x64x40xf32>) -> tensor<16x96x64x40xf32>
  %result = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%arg0, %arg1 : tensor<16x96x64x40xbf16>, tensor<40x1x1x40xbf16>)
    outs(%fill : tensor<16x96x64x40xf32>) -> tensor<16x96x64x40xf32>
  return %result : tensor<16x96x64x40xf32>
}

// IGEMM-DIRECT-LOAD-LABEL: func.func @conv_bf16_no_untuned_swizzle
// IGEMM-DIRECT-LOAD:       linalg.conv_2d_nhwc_fhwc {
// IGEMM-DIRECT-LOAD-SAME:    promotion_types = [#iree_gpu.use_global_load_dma, #iree_gpu.use_global_load_dma]

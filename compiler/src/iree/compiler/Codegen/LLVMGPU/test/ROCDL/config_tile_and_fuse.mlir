// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx942 \
// RUN: --iree-codegen-llvmgpu-early-tile-and-fuse-matmul=true --iree-codegen-llvmgpu-test-tile-and-fuse-vectorize=true \
// RUN: --iree-codegen-llvmgpu-use-igemm=false \
// RUN: --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s --check-prefix=CHECK
//
// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx942 \
// RUN: --iree-codegen-llvmgpu-use-igemm=false \
// RUN: --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s --check-prefix=LATE

// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:     --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s --check-prefix=DEFAULT

// TODO: This test is still using the legacy LLVMGPU kernel config. This needs
// to be migrated to the rocdl heuristics, but for now is just physically
// located here.

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
func.func @expanded_matmul_transpose_b(%lhs: tensor<2x64x2048xf16>, %rhs: tensor<10x64x2048xf16>) -> tensor<2x10x64x64xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<2x10x64x64xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x10x64x64xf32>) -> tensor<2x10x64x64xf32>
  %7 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
    ins(%lhs, %rhs : tensor<2x64x2048xf16>, tensor<10x64x2048xf16>) outs(%6 : tensor<2x10x64x64xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %8 = arith.extf %in : f16 to f32
    %9 = arith.extf %in_0 : f16 to f32
    %10 = arith.mulf %8, %9 : f32
    %11 = arith.addf %10, %out : f32
    linalg.yield %11 : f32
  } -> tensor<2x10x64x64xf32>
  return %7 : tensor<2x10x64x64xf32>
}

// CHECK-LABEL: func.func @expanded_matmul_transpose_b
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>

// Verify that the fill does not have the lowering config propagated to it.
//       CHECK:   linalg.fill ins

//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 0, 0, 4]
//  CHECK-SAME:     subgroup = [1, 1, 4, 1, 0]
//  CHECK-SAME:     workgroup = [1, 1, 64, 64, 0]

//        LATE:  LLVMGPUVectorDistribute

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d4, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
func.func @multi_dim_mma_schedule(%lhs: tensor<10x32x128x16xf16>, %rhs: tensor<4x32x128x16xf16>) -> tensor<10x4x32x32xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<10x4x32x32xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<10x4x32x32xf32>) -> tensor<10x4x32x32xf32>
  %7 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
    ins(%lhs, %rhs : tensor<10x32x128x16xf16>, tensor<4x32x128x16xf16>) outs(%6 : tensor<10x4x32x32xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %8 = arith.extf %in : f16 to f32
    %9 = arith.extf %in_0 : f16 to f32
    %10 = arith.mulf %8, %9 : f32
    %11 = arith.addf %10, %out : f32
    linalg.yield %11 : f32
  } -> tensor<10x4x32x32xf32>
  return %7 : tensor<10x4x32x32xf32>
}

// CHECK-LABEL: func.func @multi_dim_mma_schedule
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>

//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 0, 0, 4, 1]
//  CHECK-SAME:     subgroup = [2, 2, 1, 1, 0, 0]
//  CHECK-SAME:     workgroup = [2, 2, 32, 32, 0, 0]

//        LATE:  LLVMGPUVectorDistribute

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d3, d5, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
func.func @dynamic_multi_dim_mma_schedule(%lhs: tensor<?x6x16x?x16xf16>, %rhs: tensor<?x32x?x16xf16>) -> tensor<?x6x?x16x32xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %d0 = tensor.dim %lhs, %c0 : tensor<?x6x16x?x16xf16>
  %d2 = tensor.dim %rhs, %c0 : tensor<?x32x?x16xf16>
  %5 = tensor.empty(%d0, %d2) : tensor<?x6x?x16x32xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<?x6x?x16x32xf32>) -> tensor<?x6x?x16x32xf32>
  %7 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
    ins(%lhs, %rhs : tensor<?x6x16x?x16xf16>, tensor<?x32x?x16xf16>) outs(%6 : tensor<?x6x?x16x32xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %8 = arith.extf %in : f16 to f32
    %9 = arith.extf %in_0 : f16 to f32
    %10 = arith.mulf %8, %9 : f32
    %11 = arith.addf %10, %out : f32
    linalg.yield %11 : f32
  } -> tensor<?x6x?x16x32xf32>
  return %7 : tensor<?x6x?x16x32xf32>
}

// CHECK-LABEL: func.func @dynamic_multi_dim_mma_schedule
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>

//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 0, 0, 0, 1, 1]
//  CHECK-SAME:     subgroup = [0, 1, 0, 1, 1, 0, 0]
//  CHECK-SAME:     workgroup = [1, 2, 1, 16, 32, 0, 0]

//        LATE:  LLVMGPUVectorDistribute

// -----

func.func @mfma_matmul_1024x1024x1024(%lhs: tensor<1024x1024xf16>, %rhs: tensor<1024x1024xf16>) -> tensor<1024x1024xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %5 = tensor.empty() : tensor<1024x1024xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %7 = linalg.matmul ins(%lhs, %rhs : tensor<1024x1024xf16>, tensor<1024x1024xf16>) outs(%6 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  return %7 : tensor<1024x1024xf32>
}

// CHECK-LABEL: func.func @mfma_matmul_1024x1024x1024
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>

// Verify that the fill does not have the lowering config propagated to it.
//       CHECK:   linalg.fill ins

//       CHECK:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 2]
//  CHECK-SAME:     subgroup = [4, 4, 0]
//  CHECK-SAME:     workgroup = [128, 128, 0]

//        LATE:  LLVMGPUVectorDistribute

// -----

module {
  func.func @conv_nhwc(%3: tensor<2x258x514x768xf16>, %4: tensor<3x3x768x256xf16>) -> tensor<2x256x512x256xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %5 = tensor.empty() : tensor<2x256x512x256xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x256x512x256xf32>) -> tensor<2x256x512x256xf32>
    %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<2x258x514x768xf16>, tensor<3x3x768x256xf16>) outs(%6 : tensor<2x256x512x256xf32>) -> tensor<2x256x512x256xf32>
    return %7 : tensor<2x256x512x256xf32>
  }
}

// CHECK-LABEL: func.func @conv_nhwc
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK:   linalg.conv_2d_nhwc_hwcf {{.*}} lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     reduction = [0, 0, 0, 0, 1, 3, 4]
//  CHECK-SAME:     thread = [1, 1, 1, 1, 0, 0, 0]
//  CHECK-SAME:     workgroup = [1, 1, 1, 64, 0, 0, 0]

//        LATE:  LLVMGPUVectorDistribute

// -----

module {
  func.func @matmul_dynamic_dim(%11: tensor<?x256xf16>, %12: tensor<256x256xf16>) -> tensor<?x256xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %8 = tensor.dim %11, %c0 : tensor<?x256xf16>
    %13 = tensor.empty(%8) : tensor<?x256xf32>
    %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<?x256xf32>) -> tensor<?x256xf32>
    %15 = linalg.matmul ins(%11, %12 : tensor<?x256xf16>, tensor<256x256xf16>) outs(%14 : tensor<?x256xf32>) -> tensor<?x256xf32>
    return %15 : tensor<?x256xf32>
  }
}

// CHECK-LABEL: func.func @matmul_dynamic_dim
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK:   linalg.matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 4]
//  CHECK-SAME:     thread = [1, 4, 0]
//  CHECK-SAME:     workgroup = [1, 256, 0]

//        LATE:  LLVMGPUWarpReduction

// -----

module {
  func.func @elementwise_dynamic_dim(%11: tensor<?x256xf16>, %12: tensor<?x256xf16>) -> tensor<?x256xf16> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %8 = tensor.dim %11, %c0 : tensor<?x256xf16>
    %13 = tensor.empty(%8) : tensor<?x256xf16>
    %15 = linalg.add ins(%11, %12 : tensor<?x256xf16>, tensor<?x256xf16>) outs(%13 : tensor<?x256xf16>) -> tensor<?x256xf16>
    return %15 : tensor<?x256xf16>
  }
}

// CHECK-LABEL: func.func @elementwise_dynamic_dim
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK:   linalg.add {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     thread = [1, 4]
//  CHECK-SAME:     workgroup = [1, 256]

// -----

module @elementwise_unaligned {
  func.func @elementwise_unaligned(%11: tensor<180x180xf16>, %12: tensor<180x180xf16>) -> tensor<180x180xf16> {
    %cst = arith.constant 0.000000e+00 : f32
    %13 = tensor.empty() : tensor<180x180xf16>
    %15 = linalg.add ins(%11, %12 : tensor<180x180xf16>, tensor<180x180xf16>) outs(%13 : tensor<180x180xf16>) -> tensor<180x180xf16>
    return %15 : tensor<180x180xf16>
  }
}

// CHECK-LABEL: func.func @elementwise_unaligned
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>

// -----

module @elementwise_large_rank {
  func.func @elementwise_large_rank(%11: tensor<3x5x7x11x13x17x19x23xf16>, %12: tensor<3x5x7x11x13x17x19x23xf16>) -> tensor<3x5x7x11x13x17x19x23xf16> {
    %cst = arith.constant 0.000000e+00 : f32
    %13 = tensor.empty() : tensor<3x5x7x11x13x17x19x23xf16>
    %15 = linalg.add ins(%11, %12 : tensor<3x5x7x11x13x17x19x23xf16>, tensor<3x5x7x11x13x17x19x23xf16>) outs(%13 : tensor<3x5x7x11x13x17x19x23xf16>) -> tensor<3x5x7x11x13x17x19x23xf16>
    return %15 : tensor<3x5x7x11x13x17x19x23xf16>
  }
}

// Verify that a lowering config is set on large rank tensors with unaligned
// shapes.
// CHECK-LABEL: func.func @elementwise_large_rank
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>

// -----

module {
  func.func @multi_mma_data_tiled_unrolled_MFMA_F32_16x16x4_F32(
        %3: tensor<1x8x8x4x16x4xf32>, %4: tensor<1x8x4x2x4x16x4xf32>, %5: tensor<1x1x8x4x2x4x16x4xf32>) -> tensor<1x1x8x4x2x4x16x4xf32> {
    %c0 = arith.constant 0 : index
    %c65536 = arith.constant 65536 : index
    %c131072 = arith.constant 131072 : index
    %6 = iree_gpu.multi_mma %3, %4, %5 {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                         affine_map<(d0, d1, d2) -> (d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = [#iree_gpu.iterator_type<parallel>,
                          #iree_gpu.iterator_type<parallel>,
                          #iree_gpu.iterator_type<reduction>],
        kind = #iree_gpu.data_tiled_mma_layout<
                          intrinsic =  MFMA_F32_16x16x4_F32,
                          intrinsics_m = 8, intrinsics_n = 2,
                          subgroups_n = 4,
                          intrinsics_k = 4>}
        : tensor<1x8x8x4x16x4xf32>, tensor<1x8x4x2x4x16x4xf32> into tensor<1x1x8x4x2x4x16x4xf32>
    return %6 : tensor<1x1x8x4x2x4x16x4xf32>
  }
}

// CHECK-LABEL: func.func @multi_mma_data_tiled_unrolled_MFMA_F32_16x16x4_F32
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64
//  CHECK-SAME:   {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}
//       CHECK:   iree_gpu.multi_mma {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     reduction = [0, 0, 1]
//  CHECK-SAME:     workgroup = [1, 1, 0]

// -----

module {
func.func @unaligned_to_intrinsic_batched_matmul(%lhs : tensor<12x2x577xf32>, %rhs : tensor<12x577x577xf32>) -> tensor<12x2x577xf32> {
    %c0 = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<12x2x577xf32>
    %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<12x2x577xf32>) -> tensor<12x2x577xf32>
    %mm = linalg.batch_matmul ins(%lhs, %rhs : tensor<12x2x577xf32>, tensor<12x577x577xf32>) outs(%fill : tensor<12x2x577xf32>) -> tensor<12x2x577xf32>
    return %mm :  tensor<12x2x577xf32>
}
}

// LATE-LABEL: func.func @unaligned_to_intrinsic_batched_matmul
// LATE-SAME:    #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64
// LATE-SAME:    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}
//      LATE:    linalg.batch_matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  LATE-SAME:     padding = [1, 16, 16, 4]
//  LATE-SAME:     promote_operands = [0, 1, 2]
//  LATE-SAME:     reduction = [0, 0, 0, 1]
//  LATE-SAME:     subgroup = [0, 1, 1, 0]
//  LATE-SAME:     workgroup = [1, 16, 16, 0]

// -----

module {
func.func @unaligned_matmul_with_two_reduce_dim(%arg0: tensor<196x9x4xf32>, %arg1: tensor<9x16x4xf32>) -> tensor<196x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<196x16xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<196x16xf32>) -> tensor<196x16xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2)>], iterator_types = ["parallel", "reduction", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<196x9x4xf32>, tensor<9x16x4xf32>) outs(%1 : tensor<196x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.mulf %in, %in_0 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<196x16xf32>
  return %2 : tensor<196x16xf32>
}
}

// LATE-LABEL: func.func @unaligned_matmul_with_two_reduce_dim
// LATE-SAME:  {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64
// LATE:       linalg.generic
// LATE-SAME:  {lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>
// LATE-SAME:  padding = [16, 1, 16, 4]
// LATE-SAME:  promote_operands = [0, 1, 2]
// LATE-SAME:  reduction = [0, 1, 0, 1],
// LATE-SAME:  subgroup = [1, 0, 1, 0],
// LATE-SAME:  workgroup = [16, 0, 16, 0]}

// -----

module {
func.func @aligned_dynamic_matmul_with_two_reduce_dim(%arg0: tensor<192x?x16xf32>, %arg1: tensor<?x16x16xf32>) -> tensor<192x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<192x16xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<192x16xf32>) -> tensor<192x16xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2)>], iterator_types = ["parallel", "reduction", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<192x?x16xf32>, tensor<?x16x16xf32>) outs(%1 : tensor<192x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.mulf %in, %in_0 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<192x16xf32>
  return %2 : tensor<192x16xf32>
}
}

// CHECK-LABEL: func.func @aligned_dynamic_matmul_with_two_reduce_dim
// CHECK-SAME:  {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64
// CHECK:       linalg.generic
// CHECK-SAME:  {lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>
// CHECK-SAME:  promote_operands = [0, 1]
// CHECK-SAME:  reduction = [0, 1, 0, 4],
// CHECK-SAME:  subgroup = [2, 0, 1, 0],
// CHECK-SAME:  workgroup = [64, 0, 16, 0]}

// -----

module {
func.func @unaligned_dynamic_matmul_with_two_reduce_dim(%arg0: tensor<196x?x4xf32>, %arg1: tensor<?x16x4xf32>) -> tensor<196x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<196x16xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<196x16xf32>) -> tensor<196x16xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2)>], iterator_types = ["parallel", "reduction", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<196x?x4xf32>, tensor<?x16x4xf32>) outs(%1 : tensor<196x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.mulf %in, %in_0 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<196x16xf32>
  return %2 : tensor<196x16xf32>
}
}

// CHECK-LABEL: func.func @unaligned_dynamic_matmul_with_two_reduce_dim
// CHECK-SAME:  {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64
// CHECK:       linalg.generic
// CHECK-SAME:  promote_operands = [0, 1]
// CHECK-SAME:  reduction = [0, 4, 0, 4],
// CHECK-SAME:  thread = [1, 0, 1, 0],
// CHECK-SAME:  workgroup = [4, 0, 16, 0]}

// -----

module {
func.func @unaligned_to_intrinsic_batched_matmul_tiling_check(%lhs : tensor<12x577x577xf32>, %rhs : tensor<12x577x1024xf32>) -> tensor<12x577x1024xf32> {
    %c0 = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<12x577x1024xf32>
    %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<12x577x1024xf32>) -> tensor<12x577x1024xf32>
    %mm = linalg.batch_matmul ins(%lhs, %rhs : tensor<12x577x577xf32>, tensor<12x577x1024xf32>) outs(%fill : tensor<12x577x1024xf32>) -> tensor<12x577x1024xf32>
    return %mm :  tensor<12x577x1024xf32>
}
}

// Note this test is used to check if a tuning parameter of right size can be
// derived through deduceMMASchedule() in the case of unaligned shapes.
// For existing unaligned shapes, C promotion always happens and failure in
// considering this will severely underestimates the required shared memory.
// In this unit test, if C promotion is not considered, it will deduce a MMA
// schedule with nTileSize of 16 while in reality it should be 8.

// LATE-LABEL: func.func @unaligned_to_intrinsic_batched_matmul_tiling_check
// LATE-SAME:    #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64
// LATE-SAME:    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}
//      LATE:    linalg.batch_matmul {{.*}}lowering_config = #iree_gpu.lowering_config
//  LATE-SAME:     padding = [1, 16, 512, 4]
//  LATE-SAME:     promote_operands = [0, 1, 2]
//  LATE-SAME:     reduction = [0, 0, 0, 1]
//  LATE-SAME:     subgroup = [0, 1, 8, 0]
//  LATE-SAME:     workgroup = [1, 16, 512, 0]

// -----

func.func @large_scatter(%arg0: tensor<3x2048x2048xf32>,
                   %arg1: tensor<3x1xi32>) -> tensor<3x2048x2048xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<3x2048x2048xf32>
  %1 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%arg0, %arg1 : tensor<3x2048x2048xf32>, tensor<3x1xi32>) outs(%0 : tensor<3x2048x2048xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    iree_linalg_ext.yield %arg2 : f32
  } -> tensor<3x2048x2048xf32>
  return %1 : tensor<3x2048x2048xf32>
}

// CHECK-LABEL: func.func @large_scatter
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64

//       CHECK:   linalg_ext.scatter {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     thread = [1, 1, 4]
//  CHECK-SAME:     workgroup = [1, 1, 256]

// -----

func.func @small_scatter(%arg0: tensor<3x32x16xf32>,
                         %arg1: tensor<3x1xi32>) -> tensor<3x32x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<3x32x16xf32>
  %1 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%arg0, %arg1 : tensor<3x32x16xf32>, tensor<3x1xi32>) outs(%0 : tensor<3x32x16xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    iree_linalg_ext.yield %arg2 : f32
  } -> tensor<3x32x16xf32>
  return %1 : tensor<3x32x16xf32>
}

// CHECK-LABEL: func.func @small_scatter
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64

//       CHECK:   linalg_ext.scatter {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     thread = [1, 1, 4]
//  CHECK-SAME:     workgroup = [1, 16, 16]

// -----

func.func @smaller_scatter(%arg0: tensor<3x4x16xf32>,
                         %arg1: tensor<3x1xi32>) -> tensor<3x4x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<3x4x16xf32>
  %1 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%arg0, %arg1 : tensor<3x4x16xf32>, tensor<3x1xi32>) outs(%0 : tensor<3x4x16xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    iree_linalg_ext.yield %arg2 : f32
  } -> tensor<3x4x16xf32>
  return %1 : tensor<3x4x16xf32>
}

// CHECK-LABEL: func.func @smaller_scatter
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64

//       CHECK:   linalg_ext.scatter {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     thread = [1, 1, 4]
//  CHECK-SAME:     workgroup = [3, 2, 16]

// -----

func.func @only_scattered_dim(%arg0: tensor<48xf32>,
                              %arg1: tensor<48x2xi32>) -> tensor<100x100xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<100x100xf32>
  %1 = iree_linalg_ext.scatter dimension_map = [0, 1] unique_indices(true)
    ins(%arg0, %arg1 : tensor<48xf32>, tensor<48x2xi32>) outs(%0 : tensor<100x100xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    iree_linalg_ext.yield %arg2 : f32
  } -> tensor<100x100xf32>
  return %1 : tensor<100x100xf32>
}

// CHECK-LABEL: func.func @only_scattered_dim
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUDistribute workgroup_size = [128, 1, 1] subgroup_size = 64

//       CHECK:   linalg_ext.scatter {{.*}}lowering_config = #iree_codegen.lowering_config
//  CHECK-SAME:     tile_sizes = {{\[}}[128]]

// -----

func.func @dynamic_scatter(%arg0: tensor<3x32x?xf32>,
                           %arg1: tensor<3x1xi32>,
                           %arg2: tensor<3x32x?xf32>) -> tensor<3x32x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %1 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%arg0, %arg1 : tensor<3x32x?xf32>, tensor<3x1xi32>) outs(%arg2 : tensor<3x32x?xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):
    iree_linalg_ext.yield %arg3 : f32
  } -> tensor<3x32x?xf32>
  return %1 : tensor<3x32x?xf32>
}

// CHECK-LABEL: func.func @dynamic_scatter
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64

//       CHECK:   linalg_ext.scatter {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     thread = [1, 1, 4]
//  CHECK-SAME:     workgroup = [1, 1, 256]

// -----

func.func @elementwise_scatter(%arg0: tensor<3x2048x2048xf32>,
                               %arg1: tensor<3x2048x2048xf32>,
                               %arg2: tensor<3x1xi32>) -> tensor<3x2048x2048xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<3x2048x2048xf32>
  %1 = linalg.add ins(%arg0, %arg1 : tensor<3x2048x2048xf32>, tensor<3x2048x2048xf32>)
    outs(%0 : tensor<3x2048x2048xf32>) -> tensor<3x2048x2048xf32>
  %2 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%1, %arg2 : tensor<3x2048x2048xf32>, tensor<3x1xi32>) outs(%0 : tensor<3x2048x2048xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):
    iree_linalg_ext.yield %arg3 : f32
  } -> tensor<3x2048x2048xf32>
  return %2 : tensor<3x2048x2048xf32>
}

// CHECK-LABEL: func.func @elementwise_scatter
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64

//       CHECK:   linalg.add {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     thread = [1, 1, 4]
//  CHECK-SAME:     workgroup = [1, 1, 256]

// Verify that the scatter does not get a lowering config
//       CHECK:   linalg_ext.scatter dimension_map

// -----

func.func @scatter_as_root_op(%arg0: tensor<4x?xi64>,
                              %arg1: tensor<4x?x32x8x128xf16>) -> tensor<?x32x8x128xf16> {
  %i1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %i1 : tensor<4x?xi64>
  %1 = tensor.empty(%0) : tensor<4x?xi32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<4x?xi64>) outs(%1 : tensor<4x?xi32>) {
  ^bb0(%in: i64, %out: i32):
    %3 = arith.trunci %in : i64 to i32
    linalg.yield %3 : i32
  } -> tensor<4x?xi32>

  %4 = tensor.dim %arg1, %i1 : tensor<4x?x32x8x128xf16>
  %5 = tensor.empty(%4) : tensor<?x32x8x128xf16>
  %6 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%arg1, %2 : tensor<4x?x32x8x128xf16>, tensor<4x?xi32>) outs(%5 : tensor<?x32x8x128xf16>) {
  ^bb0(%arg2: f16, %arg3: f16):
    iree_linalg_ext.yield %arg2 : f16
  } -> tensor<?x32x8x128xf16>
  return %6 : tensor<?x32x8x128xf16>
}

// CHECK-LABEL: func.func @scatter_as_root_op

// Verify that the linalg.generic does not get a lowering config
// CHECK:      linalg.generic
// CHECK-SAME: {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: ins(%arg0 : tensor<4x?xi64>) outs({{.*}} : tensor<4x?xi32>) {

// Verify that the scatter op gets a lowering config
// CHECK:      iree_linalg_ext.scatter {{.*}}lowering_config =

// -----

func.func @set_encoding_gpu(%0 : tensor<1234x567xi8>) -> tensor<10x9x8x4x4x4x2x8xi8> {
  %c0_i8 = arith.constant 0 : i8
  %22 = tensor.empty() : tensor<10x9x128x64xi8>
  %pack = linalg.pack %0 padding_value(%c0_i8 : i8)
      outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [128, 64]
      into %22 : tensor<1234x567xi8> -> tensor<10x9x128x64xi8>
  %expanded = tensor.expand_shape %pack [[0], [1], [2, 3, 4], [5, 6, 7]]
      output_shape [10, 9, 4, 8, 4, 2, 4, 8]
      : tensor<10x9x128x64xi8> into tensor<10x9x4x8x4x2x4x8xi8>
  %23 = tensor.empty() : tensor<10x9x8x4x4x4x2x8xi8>
  %24 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d4, d2, d5, d6, d3, d7)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%expanded : tensor<10x9x4x8x4x2x4x8xi8>) outs(%23 : tensor<10x9x8x4x4x4x2x8xi8>) {
  ^bb0(%in: i8, %out: i8):
    linalg.yield %in : i8
  } -> tensor<10x9x8x4x4x4x2x8xi8>
  return %24 : tensor<10x9x8x4x4x4x2x8xi8>
}

// CHECK-LABEL: func.func @set_encoding_gpu
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     thread = [1, 1, 1, 1, 1, 1, 1, 1]
//  CHECK-SAME:     workgroup = [1, 1, 8, 4, 4, 4, 2, 8]

// -----

func.func @unset_encoding_gpu(%arg0: tensor<10x5x4x8x2x4x16x4xi32>) -> tensor<1234x567xi32> {
  %c0_i8 = arith.constant 0 : i8
  %0 = tensor.empty() : tensor<10x5x4x8x4x4x16x2xi32>
  %transposed = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d3, d7, d2, d6, d4)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<10x5x4x8x2x4x16x4xi32>) outs(%0 : tensor<10x5x4x8x4x4x16x2xi32>) {
  ^bb0(%in: i32, %out: i32):
    linalg.yield %in : i32
  } -> tensor<10x5x4x8x4x4x16x2xi32>
  %collapsed = tensor.collapse_shape %transposed [[0], [1], [2, 3, 4], [5, 6, 7]]
      : tensor<10x5x4x8x4x4x16x2xi32> into tensor<10x5x128x128xi32>
  %1 = tensor.empty() : tensor<1234x567xi32>
  %unpack = linalg.unpack %collapsed
      outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [128, 128]
      into %1 : tensor<10x5x128x128xi32> -> tensor<1234x567xi32>
  return %unpack : tensor<1234x567xi32>
}

// CHECK-LABEL: func.func @unset_encoding_gpu
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     thread = [1, 1, 1, 1, 1, 1, 1, 1]
//  CHECK-SAME:     workgroup = [1, 1, 4, 8, 4, 4, 16, 2]

// -----

func.func @pack_dynamic_producer(%arg0: tensor<?x?xi8>, %d0: index, %d1: index, %d2: index, %d3: index) -> tensor<?x?x32x32xi8> {
  %c0_i8 = arith.constant 0 : i8
  %init0 = tensor.empty(%d0, %d1) : tensor<?x?xi8>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xi8>) outs(%init0 : tensor<?x?xi8>) {
  ^bb0(%in: i8, %out: i8):
    linalg.yield %in : i8
  } -> tensor<?x?xi8>
  %init1 = tensor.empty(%d2, %d3) : tensor<?x?x32x32xi8>
  %pack = linalg.pack %0 padding_value(%c0_i8 : i8)
      outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 32]
      into %init1 : tensor<?x?xi8> -> tensor<?x?x32x32xi8>
  return %pack : tensor<?x?x32x32xi8>
}

// CHECK-LABEL: func.func @pack_dynamic_producer
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [1, 1, 1] subgroup_size = 64>
//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     thread = [1, 1]
//  CHECK-SAME:     workgroup = [32, 32]

// -----

func.func @pack_full_tile(%arg0: tensor<32x32xi8>) -> tensor<1x1x32x32xi8> {
  %c0_i8 = arith.constant 0 : i8
  %init0 = tensor.empty() : tensor<32x32xi8>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<32x32xi8>) outs(%init0 : tensor<32x32xi8>) {
  ^bb0(%in: i8, %out: i8):
    linalg.yield %in : i8
  } -> tensor<32x32xi8>
  %init1 = tensor.empty() : tensor<1x1x32x32xi8>
  %pack = linalg.pack %0 padding_value(%c0_i8 : i8)
      outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 32]
      into %init1 : tensor<32x32xi8> -> tensor<1x1x32x32xi8>
  return %pack : tensor<1x1x32x32xi8>
}

// CHECK-LABEL: func.func @pack_full_tile
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     thread = [1, 4]
//  CHECK-SAME:     workgroup = [32, 32]

// -----

func.func @pack_dynamic_tile(%arg0: tensor<32x32xi8>, %d0: index, %d1: index, %tile0: index, %tile1: index) -> tensor<?x?x?x?xi8> {
  %c0_i8 = arith.constant 0 : i8
  %init0 = tensor.empty() : tensor<32x32xi8>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<32x32xi8>) outs(%init0 : tensor<32x32xi8>) {
  ^bb0(%in: i8, %out: i8):
    linalg.yield %in : i8
  } -> tensor<32x32xi8>
  %init1 = tensor.empty(%d0, %d1, %tile0, %tile1) : tensor<?x?x?x?xi8>
  %pack = linalg.pack %0 padding_value(%c0_i8 : i8)
      outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [%tile0, %tile1]
      into %init1 : tensor<32x32xi8> -> tensor<?x?x?x?xi8>
  return %pack : tensor<?x?x?x?xi8>
}

// CHECK-LABEL: func.func @pack_dynamic_tile
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     thread = [1, 4]
//  CHECK-SAME:     workgroup = [8, 32]

// -----

func.func @single_pack(%arg0: tensor<100x250xi32>) -> tensor<16x4x16x32xi32> {
  %c42_i32 = arith.constant 42 : i32
  %c0 = arith.constant 0 : index
  %3 = tensor.empty() : tensor<16x4x16x32xi32>
  %pack = linalg.pack %arg0
      padding_value(%c42_i32 : i32)
      outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 32]
      into %3 : tensor<100x250xi32> -> tensor<16x4x16x32xi32>
  return %pack : tensor<16x4x16x32xi32>
}

// CHECK-LABEL: func.func @single_pack
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK:   linalg.pack {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     thread = [1, 1, 1, 4]
//  CHECK-SAME:     workgroup = [1, 1, 16, 32]

// -----

// Tests that we are able to compute appropriate workgroup tile sizes when
// there are multiple relayout ops in the dispatch.

func.func @unpack_pack(%arg0: tensor<8x4x32x32xi32>) -> tensor<16x4x16x32xi32> {
  %c42_i32 = arith.constant 42 : i32
  %c0 = arith.constant 0 : index
  %3 = tensor.empty() : tensor<100x250xi32>
  %unpack = linalg.unpack %arg0
      outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [32, 32]
      into %3 : tensor<8x4x32x32xi32> -> tensor<100x250xi32>
  %4 = tensor.empty() : tensor<16x4x16x32xi32>
  %pack = linalg.pack %unpack
      padding_value(%c42_i32 : i32)
      outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 32]
      into %4 : tensor<100x250xi32> -> tensor<16x4x16x32xi32>
  return %pack : tensor<16x4x16x32xi32>
}

// CHECK-LABEL: func.func @unpack_pack
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK:   linalg.pack {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     thread = [1, 1, 1, 4]
//  CHECK-SAME:     workgroup = [2, 1, 16, 32]

// -----

module {
  func.func @erf(%13 : tensor<2x1024x5120xf16>, %12 : tensor<2x1024x5120xf16>, %9 : tensor<5120xf16>, %10 : tensor<f32>) -> tensor<2x1024x5120xi8> {
    %cst = arith.constant 0.000000e+00 : f16
    %11 = tensor.empty() : tensor<2x1024x5120xi8>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%13, %12, %9, %10 : tensor<2x1024x5120xf16>, tensor<2x1024x5120xf16>, tensor<5120xf16>, tensor<f32>) outs(%11 : tensor<2x1024x5120xi8>) {
    ^bb0(%in: f16, %in_4: f16, %in_5: f16, %in_6: f32, %out: i8):
      %17 = math.erf %in : f16
      %30 = arith.fptosi %17 : f16 to i8
      linalg.yield %30 : i8
    } -> tensor<2x1024x5120xi8>
    return %14 : tensor<2x1024x5120xi8>
  }
}

// CHECK-LABEL: func.func @erf
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     thread = [1, 1, 8]
//  CHECK-SAME:     workgroup = [1, 1, 512]

// -----

func.func @map_scatter(%arg0: tensor<100x250xi32>) -> tensor<100x250xi32> {
  %0 = tensor.empty() : tensor<100x250xi32>
  %1 = iree_linalg_ext.map_scatter %arg0 into %0 {
  ^bb0(%arg1: index, %arg2: index):
    %true = arith.constant true
    iree_linalg_ext.yield %arg1, %arg2, %true : index, index, i1
  } : tensor<100x250xi32> into tensor<100x250xi32> -> tensor<100x250xi32>
  return %1 : tensor<100x250xi32>
}

// CHECK-LABEL: func.func @map_scatter
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK:   iree_linalg_ext.map_scatter {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     thread = [1, 1]
//  CHECK-SAME:     workgroup = [1, 64]

// -----

func.func @small_reduction(%arg0 : tensor<2x?xf32>, %arg1 : tensor<?xf32>, %arg2 : index) -> tensor<?xf32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<2x?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32) :
      %1 = arith.addf %b0, %b1 : f32
      linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
// DEFAULT-LABEL: @small_reduction
//  DEFAULT-SAME:     #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse

// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx942 \
// RUN: --iree-codegen-llvmgpu-use-igemm=false --iree-codegen-llvmgpu-use-direct-convolution=true --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
func.func @conv_nhwc_fhwc(%arg0: tensor<16x50x34x576xf16>, %arg1: tensor<576x3x3x576xf16>, %arg2: tensor<16x48x32x576xf32>) -> tensor<16x48x32x576xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x50x34x576xf16>, tensor<576x3x3x576xf16>) outs(%arg2 : tensor<16x48x32x576xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %1 = arith.extf %in : f16 to f32
    %2 = arith.extf %in_0 : f16 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<16x48x32x576xf32>
  return %0 : tensor<16x48x32x576xf32>
}

// CHECK-LABEL: func.func @conv_nhwc_fhwc
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = false
//  CHECK-SAME:   use_igemm_convolution = false

//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>

//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 0, 0, 1, 1, 4]
//  CHECK-SAME:     subgroup = [0, 0, 1, 1, 0, 0, 0]
//  CHECK-SAME:     workgroup = [1, 1, 32, 64, 0, 0, 0]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
func.func @conv_nhwc_fhwc_unaligned(%arg0: tensor<16x26x19x287xf16>, %arg1: tensor<287x3x3x287xf16>, %arg2: tensor<16x24x17x287xf32>) -> tensor<16x24x17x287xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x26x19x287xf16>, tensor<287x3x3x287xf16>) outs(%arg2 : tensor<16x24x17x287xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %1 = arith.extf %in : f16 to f32
    %2 = arith.extf %in_0 : f16 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<16x24x17x287xf32>
  return %0 : tensor<16x24x17x287xf32>
}

// CHECK-LABEL: func.func @conv_nhwc_fhwc_unaligned
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = false
//  CHECK-SAME:   use_igemm_convolution = false

//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>

//  CHECK-SAME:     padding_conv = [1, 1, 32, 32, 0, 0, 32]
//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 0, 0, 1, 1, 2]
//  CHECK-SAME:     subgroup = [0, 0, 1, 1, 0, 0, 0]
//  CHECK-SAME:     workgroup = [1, 1, 32, 32, 0, 0, 0]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0 + d4, d1 + d5, d2, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d3, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
func.func @group_conv_hwgc_gfhwc_unaligned(%arg0: tensor<61x93x16x56xbf16>, %arg1: tensor<16x56x3x3x56xbf16>, %arg2: tensor<59x91x16x56xf32>) -> tensor<59x91x16x56xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<61x93x16x56xbf16>, tensor<16x56x3x3x56xbf16>) outs(%arg2 : tensor<59x91x16x56xf32>) {
    ^bb0(%in: bf16, %in_4: bf16, %out: f32):
      %10 = arith.extf %in : bf16 to f32
      %11 = arith.extf %in_4 : bf16 to f32
      %12 = arith.mulf %10, %11 : f32
      %13 = arith.addf %out, %12 : f32
      linalg.yield %13 : f32
    } -> tensor<59x91x16x56xf32>
  return %0 : tensor<59x91x16x56xf32>
}

// CHECK-LABEL: func.func @group_conv_hwgc_gfhwc_unaligned
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = false
//  CHECK-SAME:   use_igemm_convolution = false

//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_BF16>
//  CHECK-SAME:     padding_conv = [1, 32, 1, 64, 0, 0, 8]
//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 0, 0, 1, 1, 1]
//  CHECK-SAME:     subgroup = [0, 1, 0, 1, 0, 0, 0]
//  CHECK-SAME:     workgroup = [1, 32, 1, 64, 0, 0, 0]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d4, d2, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
func.func @conv_nhwc_fhc_two_batch_dims(%arg0: tensor<16x50x32x576xf16>, %arg1: tensor<576x3x576xf16>, %arg2: tensor<16x48x32x576xf32>) -> tensor<16x48x32x576xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x50x32x576xf16>, tensor<576x3x576xf16>) outs(%arg2 : tensor<16x48x32x576xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %1 = arith.extf %in : f16 to f32
    %2 = arith.extf %in_0 : f16 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<16x48x32x576xf32>
  return %0 : tensor<16x48x32x576xf32>
}

// CHECK-LABEL: func.func @conv_nhwc_fhc_two_batch_dims
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = false
//  CHECK-SAME:   use_igemm_convolution = false

//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>

//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 0, 0, 1, 4]
//  CHECK-SAME:     subgroup = [0, 0, 1, 1, 0, 0]
//  CHECK-SAME:     workgroup = [1, 1, 32, 64, 0, 0]

// -----

func.func @conv_nchw_named(%3: tensor<2x128x34x34xf32>, %4: tensor<64x128x3x3xf32>) -> tensor<2x64x32x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<2x64x32x32xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x64x32x32xf32>) -> tensor<2x64x32x32xf32>
  %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<2x128x34x34xf32>, tensor<64x128x3x3xf32>) outs(%6 : tensor<2x64x32x32xf32>) -> tensor<2x64x32x32xf32>
  return %7 : tensor<2x64x32x32xf32>
}

// CHECK-LABEL: func.func @conv_nchw_named
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = false
//  CHECK-SAME:   use_igemm_convolution = false

//       CHECK:   linalg.conv_2d_nchw_fchw {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>

//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 0, 0, 16, 1, 1]
//  CHECK-SAME:     subgroup = [0, 1, 0, 1, 0, 0, 0]
//  CHECK-SAME:     workgroup = [1, 64, 1, 32, 0, 0, 0]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1 + d5, d2 + d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
func.func @conv_chwn_fhwn(%arg0: tensor<16x26x18x144xf16>, %arg1: tensor<16x24x16x144xf16>, %arg2: tensor<144x3x3x144xf32>) -> tensor<144x3x3x144xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x26x18x144xf16>, tensor<16x24x16x144xf16>) outs(%arg2 : tensor<144x3x3x144xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %1 = arith.extf %in : f16 to f32
    %2 = arith.extf %in_0 : f16 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<144x3x3x144xf32>
  return %0 : tensor<144x3x3x144xf32>
}

// CHECK-LABEL: func.func @conv_chwn_fhwn
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64
//  CHECK-SAME:   #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = false
//  CHECK-SAME:   use_igemm_convolution = false

//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>

//  CHECK-SAME:     promote_operands = [0, 1]
//  CHECK-SAME:     reduction = [0, 0, 0, 0, 1, 1, 1]
//  CHECK-SAME:     subgroup = [1, 0, 0, 1, 0, 0, 0]
//  CHECK-SAME:     workgroup = [16, 1, 1, 16, 0, 0, 0]

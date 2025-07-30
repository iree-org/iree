// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-pad-convs))" | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, padding_conv = [1, 8, 32, 32, 0, 0, 32]}>
func.func @conv_2d_nhwc_fhwc(%arg0: tensor<16x26x19x287xf16>, %arg1: tensor<287x3x3x287xf16>, %arg2: tensor<16x24x17x287xf32>) -> tensor<16x24x17x287xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x26x19x287xf16>, tensor<287x3x3x287xf16>) outs(%arg2 : tensor<16x24x17x287xf32>) attrs = {lowering_config = #lowering_config} {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %1 = arith.extf %in : f16 to f32
    %2 = arith.extf %in_0 : f16 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<16x24x17x287xf32>
  return %0 : tensor<16x24x17x287xf32>
}

// CHECK-LABEL: func.func @conv_2d_nhwc_fhwc
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: tensor<16x26x19x287xf16>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: tensor<287x3x3x287xf16>
//  CHECK-SAME:   %[[C:[A-Za-z0-9]+]]: tensor<16x24x17x287xf32>
//       CHECK:   %[[PADDED_LHS:.+]] = tensor.pad %[[A]] low[0, 0, 0, 0] high[0, 0, 15, 1]
//       CHECK:   %[[PADDED_RHS:.+]] = tensor.pad %[[B]] low[0, 0, 0, 0] high[1, 0, 0, 1]
//       CHECK:   %[[PADDED_INIT:.+]] = tensor.pad %[[C]] low[0, 0, 0, 0] high[0, 0, 15, 1]
//       CHECK:   %[[PADDED_RESULT:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[PADDED_LHS]], %[[PADDED_RHS]] : tensor<16x26x34x288xf16>, tensor<288x3x3x288xf16>)
//  CHECK-SAME:     outs(%[[PADDED_INIT]] : tensor<16x24x32x288xf32>)
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[PADDED_RESULT]][0, 0, 0, 0] [16, 24, 17, 287] [1, 1, 1, 1]
//  CHECK-SAME:     : tensor<16x24x32x288xf32> to tensor<16x24x17x287xf32>
//       CHECK:   return %[[EXTRACT]] : tensor<16x24x17x287xf32>

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1 + d5 * 2, d2 + d6 * 2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, padding_conv = [16, 1, 1, 16, 0, 0, 0]}>
func.func @conv_2d_chwn_chwf(%arg0: tensor<16x193x129x40xbf16>, %arg1: tensor<16x96x64x40xbf16>, %arg2: tensor<40x3x3x40xf32>) -> tensor<40x3x3x40xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x193x129x40xbf16>, tensor<16x96x64x40xbf16>) outs(%arg2 : tensor<40x3x3x40xf32>) attrs =  {lowering_config = #lowering_config} {
  ^bb0(%in: bf16, %in_0: bf16, %out: f32):
    %1 = arith.extf %in : bf16 to f32
    %2 = arith.extf %in_0 : bf16 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<40x3x3x40xf32>
  return %0 : tensor<40x3x3x40xf32>
}

// CHECK-LABEL: func.func @conv_2d_chwn_chwf
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: tensor<16x193x129x40xbf16>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: tensor<16x96x64x40xbf16>
//  CHECK-SAME:   %[[C:[A-Za-z0-9]+]]: tensor<40x3x3x40xf32>
//       CHECK:   %[[PADDED_LHS:.+]] = tensor.pad %[[A]] low[0, 0, 0, 0] high[0, 0, 0, 8]
//       CHECK:   %[[PADDED_RHS:.+]] = tensor.pad %[[B]] low[0, 0, 0, 0] high[0, 0, 0, 8]
//       CHECK:   %[[PADDED_INIT:.+]] = tensor.pad %[[C]] low[0, 0, 0, 0] high[8, 0, 0, 8]
//       CHECK:   %[[PADDED_RESULT:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[PADDED_LHS]], %[[PADDED_RHS]] : tensor<16x193x129x48xbf16>, tensor<16x96x64x48xbf16>)
//  CHECK-SAME:     outs(%[[PADDED_INIT]] : tensor<48x3x3x48xf32>)
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[PADDED_RESULT]][0, 0, 0, 0] [40, 3, 3, 40] [1, 1, 1, 1]
//  CHECK-SAME:     : tensor<48x3x3x48xf32> to tensor<40x3x3x40xf32>
//       CHECK:   return %[[EXTRACT]] : tensor<40x3x3x40xf32>

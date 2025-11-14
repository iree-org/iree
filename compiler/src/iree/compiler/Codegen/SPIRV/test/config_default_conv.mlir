// RUN: iree-opt --split-input-file --iree-gpu-test-target=vp_android_baseline_2022@vulkan --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

// Convolution with consumer pointwise ops.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @nhwc_conv_pointwise_112x112x32(%4: tensor<1x112x112x32xf32>, %6: tensor<1x225x225x3xf32>, %7: tensor<3x3x3x32xf32>) -> tensor<1x112x112x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<1x112x112x32xf32>
  %8 = tensor.empty() : tensor<1x112x112x32xf32>
  %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  %10 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%6, %7 : tensor<1x225x225x3xf32>, tensor<3x3x3x32xf32>) outs(%9 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10, %4 : tensor<1x112x112x32xf32>, tensor<1x112x112x32xf32>) outs(%5 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %12 = arith.subf %in, %in_0 : f32
    linalg.yield %12 : f32
  } -> tensor<1x112x112x32xf32>
  return %11 : tensor<1x112x112x32xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 4, 4, 32], [1, 2, 2, 4], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [8, 2, 2]>
//      CHECK: func.func @nhwc_conv_pointwise_112x112x32(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

func.func @nchw_conv_2x1280x8x8(%3: tensor<2x1280x10x10xf32>, %4: tensor<1280x1280x3x3xf32>, %5: tensor<2x1280x8x8xf32>) -> tensor<2x1280x8x8xf32> {
  %6 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<2x1280x10x10xf32>, tensor<1280x1280x3x3xf32>) outs(%5 : tensor<2x1280x8x8xf32>) -> tensor<2x1280x8x8xf32>
  return %6 : tensor<2x1280x8x8xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 16, 8, 8], [1, 8, 1, 4], [0, 0, 0, 0, 4, 1, 1], [0, 0, 1, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [2, 8, 2]>
//      CHECK: func.func @nchw_conv_2x1280x8x8(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.conv_2d_nchw_fchw
// CHECK-SAME:       lowering_config = #[[CONFIG]]

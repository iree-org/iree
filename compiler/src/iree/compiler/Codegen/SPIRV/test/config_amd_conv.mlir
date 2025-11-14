// RUN: iree-opt --split-input-file --iree-gpu-test-target=rdna2@vulkan --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @nhwc_conv_pointwise_2x64x64x320(%4: tensor<2x66x66x320xf16>, %5: tensor<3x3x320x320xf16>, %6: tensor<2x64x64x320xf16>) -> tensor<2x64x64x320xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %7 = tensor.empty() : tensor<2x64x64x320xf16>
  %8 = linalg.fill ins(%cst : f16) outs(%7 : tensor<2x64x64x320xf16>) -> tensor<2x64x64x320xf16>
  %9 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%4, %5 : tensor<2x66x66x320xf16>, tensor<3x3x320x320xf16>) outs(%8 : tensor<2x64x64x320xf16>) -> tensor<2x64x64x320xf16>
  %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %6 : tensor<2x64x64x320xf16>, tensor<2x64x64x320xf16>) outs(%7 : tensor<2x64x64x320xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %11 = arith.divf %in, %in_0 : f16
    linalg.yield %11 : f16
  } -> tensor<2x64x64x320xf16>
  return %10 : tensor<2x64x64x320xf16>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 4, 4, 64], [1, 2, 2, 8], [0, 0, 0, 0, 1, 1, 8], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [8, 2, 2]>
//      CHECK: func.func @nhwc_conv_pointwise_2x64x64x320(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:     lowering_config = #[[CONFIG]]

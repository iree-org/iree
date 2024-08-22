// RUN: iree-opt --split-input-file --iree-gpu-test-target=vp_android_baseline_2022@vulkan --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

// Convolution with consumer pointwise ops.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @nhwc_conv_pointwise_112x112x32() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c112 = arith.constant 112 : index
  %c32 = arith.constant 32 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<1x112x112x32xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<1x225x225x3xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<readonly:tensor<3x3x3x32xf32>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>>
  %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x112x112x32xf32>> -> tensor<1x112x112x32xf32>
  %5 = tensor.empty() : tensor<1x112x112x32xf32>
  %6 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [1, 225, 225, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x225x225x3xf32>> -> tensor<1x225x225x3xf32>
  %7 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [3, 3, 3, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x3x32xf32>> -> tensor<3x3x3x32xf32>
  %8 = tensor.empty() : tensor<1x112x112x32xf32>
  %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  %10 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%6, %7 : tensor<1x225x225x3xf32>, tensor<3x3x3x32xf32>) outs(%9 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10, %4 : tensor<1x112x112x32xf32>, tensor<1x112x112x32xf32>) outs(%5 : tensor<1x112x112x32xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %12 = arith.subf %in, %in_0 : f32
    linalg.yield %12 : f32
  } -> tensor<1x112x112x32xf32>
  flow.dispatch.tensor.store %11, %3, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 32], strides = [1, 1, 1, 1] : tensor<1x112x112x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>>
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 4, 4, 32], [1, 2, 2, 4], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [8, 2, 2]>
//      CHECK: func.func @nhwc_conv_pointwise_112x112x32()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @nchw_conv_2x1280x8x8() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x1280x10x10xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1280x1280x3x3xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<2x1280x8x8xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 1280, 10, 10], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x1280x10x10xf32>> -> tensor<2x1280x10x10xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [1280, 1280, 3, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1280x1280x3x3xf32>> -> tensor<1280x1280x3x3xf32>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [2, 1280, 8, 8], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<2x1280x8x8xf32>> -> tensor<2x1280x8x8xf32>
  %6 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<2x1280x10x10xf32>, tensor<1280x1280x3x3xf32>) outs(%5 : tensor<2x1280x8x8xf32>) -> tensor<2x1280x8x8xf32>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0, 0], sizes = [2, 1280, 8, 8], strides = [1, 1, 1, 1] : tensor<2x1280x8x8xf32> -> !flow.dispatch.tensor<readwrite:tensor<2x1280x8x8xf32>>
  return
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 16, 8, 8], [1, 8, 1, 4], [0, 0, 0, 0, 4, 1, 1], [0, 0, 1, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [2, 8, 2]>
//      CHECK: func.func @nchw_conv_2x1280x8x8()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.conv_2d_nchw_fchw
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// RUN: iree-opt --split-input-file --iree-gpu-test-target=adreno --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

// Conv - large OC - distribute to only one workgroup dimension.

func.func @conv_112x112x512(%3: tensor<1x225x225x3xf32>, %4: tensor<3x3x3x512xf32>) -> tensor<1x112x112x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<1x112x112x512xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x112x112x512xf32>) -> tensor<1x112x112x512xf32>
  %7 = linalg.conv_2d_nhwc_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x225x225x3xf32>, tensor<3x3x3x512xf32>) outs(%6 : tensor<1x112x112x512xf32>) -> tensor<1x112x112x512xf32>
  return %7 : tensor<1x112x112x512xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 1, 8, 256], [1, 1, 8, 4], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [64, 1, 1]>
//      CHECK: func.func @conv_112x112x512(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Conv - medium OC/OW/OH - distribute to two workgroup dimensions.

func.func @conv_112x112x32(%3: tensor<1x225x225x3xf32>, %4: tensor<3x3x3x32xf32>) -> tensor<1x112x112x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<1x112x112x32xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  %7 = linalg.conv_2d_nhwc_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x225x225x3xf32>, tensor<3x3x3x32xf32>) outs(%6 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  return %7 : tensor<1x112x112x32xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 4, 16, 32], [1, 4, 2, 4], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [8, 8, 1]>
//      CHECK: func.func @conv_112x112x32(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Conv - small OC/OW/OH - distribute to all three workgroup dimensions.

func.func @conv_16x16x16(%3: tensor<1x33x33x3xf32>, %4: tensor<3x3x3x16xf32>) -> tensor<1x16x16x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<1x16x16x16xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  %7 = linalg.conv_2d_nhwc_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x33x33x3xf32>, tensor<3x3x3x16xf32>) outs(%6 : tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  return %7 : tensor<1x16x16x16xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 8, 8, 16], [1, 2, 2, 4], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [4, 4, 4]>
//      CHECK: func.func @conv_16x16x16(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Depthwise conv - small OC/OW/OH - distribute to all three workgroup dimensions.

func.func @dwconv_28x28x144(%3: tensor<1x57x57x144xf32>, %4: tensor<3x3x144xf32>) -> tensor<1x28x28x144xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<1x28x28x144xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x28x28x144xf32>) -> tensor<1x28x28x144xf32>
  %7 = linalg.depthwise_conv_2d_nhwc_hwc {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x57x57x144xf32>, tensor<3x3x144xf32>) outs(%6 : tensor<1x28x28x144xf32>) -> tensor<1x28x28x144xf32>
  return %7 : tensor<1x28x28x144xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 4, 4, 16], [1, 1, 1, 4], [0, 0, 0, 0, 1, 1], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [4, 4, 4]>
//      CHECK: func.func @dwconv_28x28x144(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Depthwise conv - tiny OC/OW/OH - starving the GPU.

func.func @dwconv_4x4x8(%3: tensor<1x9x9x8xf32>, %4: tensor<3x3x8xf32>) -> tensor<1x4x4x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<1x4x4x8xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
  %7 = linalg.depthwise_conv_2d_nhwc_hwc {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x9x9x8xf32>, tensor<3x3x8xf32>) outs(%6 : tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
  return %7 : tensor<1x4x4x8xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 4, 4, 8], [1, 1, 1, 4], [0, 0, 0, 0, 1, 1], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [2, 4, 4]>
//      CHECK: func.func @dwconv_4x4x8(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:     lowering_config = #[[CONFIG]]

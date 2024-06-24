// RUN: iree-opt --split-input-file --iree-gpu-test-target=valhall1 --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

// Conv - large OC - distribute to only one workgroup dimension.

module {
  func.func @conv_112x112x512() {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c112 = arith.constant 112 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x225x225x3xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x3x3x512xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x512xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 225, 225, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x225x225x3xf32>> -> tensor<1x225x225x3xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 3, 512], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x3x512xf32>> -> tensor<3x3x3x512xf32>
    %5 = tensor.empty() : tensor<1x112x112x512xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x112x112x512xf32>) -> tensor<1x112x112x512xf32>
    %7 = linalg.conv_2d_nhwc_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x225x225x3xf32>, tensor<3x3x3x512xf32>) outs(%6 : tensor<1x112x112x512xf32>) -> tensor<1x112x112x512xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 512], strides = [1, 1, 1, 1] : tensor<1x112x112x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x512xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 1, 4, 64], [1, 1, 4, 4], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [16, 1, 1]>
//      CHECK: func.func @conv_112x112x512()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// Conv - medium OC/OW/OH - distribute to two workgroup dimensions.

module {
  func.func @conv_112x112x32() {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c112 = arith.constant 112 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x225x225x3xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x3x3x32xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 225, 225, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x225x225x3xf32>> -> tensor<1x225x225x3xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 3, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x3x32xf32>> -> tensor<3x3x3x32xf32>
    %5 = tensor.empty() : tensor<1x112x112x32xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
    %7 = linalg.conv_2d_nhwc_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x225x225x3xf32>, tensor<3x3x3x32xf32>) outs(%6 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 32], strides = [1, 1, 1, 1] : tensor<1x112x112x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x32xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 1, 8, 32], [1, 1, 4, 4], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [8, 2, 1]>
//      CHECK: func.func @conv_112x112x32()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// Conv - small OC/OW/OH - distribute to all three workgroup dimensions.

module {
  func.func @conv_16x16x16() {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x33x33x3xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x3x3x16xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x16x16x16xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 33, 33, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x33x33x3xf32>> -> tensor<1x33x33x3xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 3, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x3x16xf32>> -> tensor<3x3x3x16xf32>
    %5 = tensor.empty() : tensor<1x16x16x16xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
    %7 = linalg.conv_2d_nhwc_hwcf {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x33x33x3xf32>, tensor<3x3x3x16xf32>) outs(%6 : tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 16, 16, 16], strides = [1, 1, 1, 1] : tensor<1x16x16x16xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x16x16x16xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 4, 4, 16], [1, 2, 2, 4], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [4, 2, 2]>
//      CHECK: func.func @conv_16x16x16()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// Depthwise conv - small OC/OW/OH - distribute to all three workgroup dimensions.

module {
  func.func @dwconv_28x28x144() {
    %c0 = arith.constant 0 : index
    %c144 = arith.constant 144 : index
    %c28 = arith.constant 28 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x57x57x144xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x3x144xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x28x28x144xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [0, 57, 57, 144], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x57x57x144xf32>> -> tensor<1x57x57x144xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [3, 3, 144], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x144xf32>> -> tensor<3x3x144xf32>
    %5 = tensor.empty() : tensor<1x28x28x144xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x28x28x144xf32>) -> tensor<1x28x28x144xf32>
    %7 = linalg.depthwise_conv_2d_nhwc_hwc {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x57x57x144xf32>, tensor<3x3x144xf32>) outs(%6 : tensor<1x28x28x144xf32>) -> tensor<1x28x28x144xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 28, 28, 144], strides = [1, 1, 1, 1] : tensor<1x28x28x144xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x28x28x144xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 4, 4, 16], [1, 2, 2, 4], [0, 0, 0, 0, 1, 1], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [4, 2, 2]>
//      CHECK: func.func @dwconv_28x28x144()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// Depthwise conv - tiny OC/OW/OH - starving the GPU.

module {
  func.func @dwconv_1x2x8() {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x3x5x8xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3x3x8xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x1x2x8xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 3, 5, 8], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x3x5x8xf32>> -> tensor<1x3x5x8xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [3, 3, 8], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x8xf32>> -> tensor<3x3x8xf32>
    %5 = tensor.empty() : tensor<1x1x2x8xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x1x2x8xf32>) -> tensor<1x1x2x8xf32>
    %7 = linalg.depthwise_conv_2d_nhwc_hwc {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x3x5x8xf32>, tensor<3x3x8xf32>) outs(%6 : tensor<1x1x2x8xf32>) -> tensor<1x1x2x8xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 1, 2, 8], strides = [1, 1, 1, 1] : tensor<1x1x2x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x1x2x8xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 1, 2, 8], [1, 1, 1, 4], [0, 0, 0, 0, 1, 1], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [2, 2, 1]>
//      CHECK: func.func @dwconv_1x2x8()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:       lowering_config = #[[CONFIG]]

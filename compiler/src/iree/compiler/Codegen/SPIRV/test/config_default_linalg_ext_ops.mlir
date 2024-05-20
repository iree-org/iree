// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 512, max_compute_workgroup_size = [512, 512, 512], subgroup_size = 16>>}>
module {
  func.func @static_1d_sort() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<1000xi32>>
    %1 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [1000], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<1000xi32>> -> tensor<1000xi32>
    %2 = iree_linalg_ext.sort {__internal_linalg_transform__ = "workgroup"} dimension(0) outs(%1 : tensor<1000xi32>) {
    ^bb0(%arg0: i32, %arg1: i32):
      %3 = arith.cmpi slt, %arg0, %arg1 : i32
      iree_linalg_ext.yield %3 : i1
    } -> tensor<1000xi32>
    flow.dispatch.tensor.store %2, %0, offsets = [0], sizes = [1000], strides = [1] : tensor<1000xi32> -> !flow.dispatch.tensor<readwrite:tensor<1000xi32>>
    return
  }
}

// Check that the workgroup count and size are (1, 1, 1) for serializing the computation.

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = []>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute workgroup_size = [1, 1, 1]>
//       CHECK: func.func @static_1d_sort()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   iree_linalg_ext.sort
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----
#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 512, max_compute_workgroup_size = [512, 512, 512], subgroup_size = 16>>}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @static_3d_sort() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<64x32x128xi32>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<64x32x128xi32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : memref<64x32x128xi32>) outs(%1 : memref<64x32x128xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    }
    iree_linalg_ext.sort {__internal_linalg_transform__ = "workgroup"} dimension(1) outs(%1 : memref<64x32x128xi32>) {
    ^bb0(%arg0: i32, %arg1: i32):
      %2 = arith.cmpi slt, %arg0, %arg1 : i32
      iree_linalg_ext.yield %2 : i1
    }
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 0, 16], [1, 0, 1]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute workgroup_size = [16, 1, 1]>
//      CHECK: func.func @static_3d_sort()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   iree_linalg_ext.sort
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----
#executable_target_vulkan_spirvfb = #hal.executable.target<"vulkan-spirv", "vulkan-spirvfb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 512, max_compute_workgroup_size = [512, 512, 512], subgroup_size = 16>>}>
module {
  func.func @static_1d_fft_stage2() attributes {hal.executable.target = #executable_target_vulkan_spirvfb} {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant dense<[1.000000e+00, 6.12323426E-17]> : tensor<2xf32>
    %cst_0 = arith.constant dense<[-0.000000e+00, -1.000000e+00]> : tensor<2xf32>
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<32xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<32xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<32xf32>> -> tensor<32xf32>
    %3 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<32xf32>> -> tensor<32xf32>
    %4:2 = iree_linalg_ext.fft {__internal_linalg_transform__ = "workgroup"} ins(%c2, %cst, %cst_0 : index, tensor<2xf32>, tensor<2xf32>) outs(%2, %3 : tensor<32xf32>, tensor<32xf32>) : tensor<32xf32>, tensor<32xf32>
    flow.dispatch.tensor.store %4#0, %0, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:tensor<32xf32>>
    flow.dispatch.tensor.store %4#1, %1, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:tensor<32xf32>>
    return
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[4]{{\]}}>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute workgroup_size = [16, 1, 1]>
//       CHECK: func.func @static_1d_fft_stage2()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----
#executable_target_vulkan_spirvfb = #hal.executable.target<"vulkan-spirv", "vulkan-spirvfb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 512, max_compute_workgroup_size = [512, 512, 512], subgroup_size = 16>>}>
module {
  func.func @static_3d_fft_stage3() attributes {hal.executable.target = #executable_target_vulkan_spirvfb} {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant dense<[1.000000e+00, 0.707106769, 6.12323426E-17, -0.707106769]> : tensor<4xf32>
    %cst_0 = arith.constant dense<[-0.000000e+00, -0.707106769, -1.000000e+00, -0.707106769]> : tensor<4xf32>
    %0 = bufferization.to_memref %cst_0 : memref<4xf32>
    %1 = bufferization.to_memref %cst : memref<4xf32>
    %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<64x128x32xf32>
    %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<64x128x32xf32>
    iree_linalg_ext.fft {__internal_linalg_transform__ = "workgroup"} ins(%c3, %1, %0 : index, memref<4xf32>, memref<4xf32>) outs(%2, %3 : memref<64x128x32xf32>, memref<64x128x32xf32>)
    return
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1, 8]{{\]}}>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute workgroup_size = [16, 1, 1]>
//       CHECK: func.func @static_3d_fft_stage3()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----
#executable_target_vulkan_spirvfb = #hal.executable.target<"vulkan-spirv", "vulkan-spirvfb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 512, max_compute_workgroup_size = [512, 512, 512], subgroup_size = 16>>}>
module {
    func.func @winograd_input_transform() attributes {hal.executable.target = #executable_target_vulkan_spirvfb} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x34x34x128xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x8x2x6x6x128xf16>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 34, 34, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x34x34x128xf16>> -> tensor<2x34x34x128xf16>
    %3 = tensor.empty() : tensor<8x8x2x6x6x128xf16>
    %4 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%2 : tensor<2x34x34x128xf16>) outs(%3 : tensor<8x8x2x6x6x128xf16>) -> tensor<8x8x2x6x6x128xf16>
    flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0, 0, 0], sizes = [8, 8, 2, 6, 6, 128], strides = [1, 1, 1, 1, 1, 1] : tensor<8x8x2x6x6x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<8x8x2x6x6x128xf16>>
    return
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 1, 0, 0, 32], [0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVWinogradVectorize workgroup_size = [32, 4, 4]>
//       CHECK: func.func @winograd_input_transform()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   iree_linalg_ext.winograd.input_transform
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----
#executable_target_vulkan_spirvfb = #hal.executable.target<"vulkan-spirv", "vulkan-spirvfb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 512, max_compute_workgroup_size = [512, 512, 512], subgroup_size = 16>>}>
module {
    func.func @winograd_output_transform() attributes {hal.executable.target = #executable_target_vulkan_spirvfb} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x8x2x6x6x128xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x36x36x128xf16>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0, 0, 0], sizes = [8, 8, 2, 6, 6, 128], strides = [1, 1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<8x8x2x6x6x128xf16>> -> tensor<8x8x2x6x6x128xf16>
    %3 = tensor.empty() : tensor<2x36x36x128xf16>
    %4 = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%2 : tensor<8x8x2x6x6x128xf16>) outs(%3 : tensor<2x36x36x128xf16>) -> tensor<2x36x36x128xf16>
    flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [2, 36, 36, 128], strides = [1, 1, 1, 1] : tensor<2x36x36x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x36x36x128xf16>>
    return
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 1, 0, 0, 32], [0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVWinogradVectorize workgroup_size = [32, 4, 4]>
//       CHECK: func.func @winograd_output_transform()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   iree_linalg_ext.winograd.output_transform
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

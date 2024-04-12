// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniformShuffle], []>, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 512, max_compute_workgroup_size = [512, 512, 512], subgroup_size = 16>>}>
#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @complex_view_as_real() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1xi32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x50xcomplex<f32>>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1x32x50x2xf32>>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x50x2xf32>>
    %4 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [1], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1xi32>> -> tensor<1xi32>
    %5 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0, 0], sizes = [1, 1, 32, 50, 2], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x32x50x2xf32>> -> tensor<1x1x32x50x2xf32>
    %6 = tensor.empty() : tensor<32x50x2xf32>
    %extracted = tensor.extract %4[%c0] : tensor<1xi32>
    %7 = arith.extsi %extracted : i32 to i64
    %8 = arith.index_cast %7 : i64 to index
    %9 = flow.dispatch.tensor.load %1, offsets = [%8, 0], sizes = [1, 50], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x50xcomplex<f32>>> -> tensor<50xcomplex<f32>>
    %10 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9 : tensor<50xcomplex<f32>>) outs(%6 : tensor<32x50x2xf32>) {
    ^bb0(%in: complex<f32>, %out: f32):
      %11 = linalg.index 0 : index
      %12 = linalg.index 1 : index
      %extracted_0 = tensor.extract %5[%c0, %c0, %11, %12, %c0] : tensor<1x1x32x50x2xf32>
      %extracted_1 = tensor.extract %5[%c0, %c0, %11, %12, %c1] : tensor<1x1x32x50x2xf32>
      %13 = complex.create %extracted_0, %extracted_1 : complex<f32>
      %14 = complex.mul %13, %in : complex<f32>
      %15 = complex.re %14 : complex<f32>
      %16 = complex.im %14 : complex<f32>
      %17 = linalg.index 2 : index
      %18 = arith.cmpi eq, %17, %c0 : index
      %19 = arith.select %18, %15, %16 : f32
      linalg.yield %19 : f32
    } -> tensor<32x50x2xf32>
    flow.dispatch.tensor.store %10, %3, offsets = [0, 0, 0], sizes = [32, 50, 2], strides = [1, 1, 1] : tensor<32x50x2xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x50x2xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[4, 2, 2], [1, 1, 1]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute workgroup_size = [2, 2, 4]>
//      CHECK: func.func @complex_view_as_real()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

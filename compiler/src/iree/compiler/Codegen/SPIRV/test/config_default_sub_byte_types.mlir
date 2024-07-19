// RUN: iree-opt --split-input-file --iree-gpu-test-target=vp_android_baseline_2022@vulkan --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @i4_dequant() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<131072x128xi4>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<131072xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<131072xf32>>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<131072x128xf32>>
    %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [131072, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<131072x128xi4>> -> tensor<131072x128xi4>
    %5 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [131072], strides = [1] : !flow.dispatch.tensor<readonly:tensor<131072xf32>> -> tensor<131072xf32>
    %6 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [131072], strides = [1] : !flow.dispatch.tensor<readonly:tensor<131072xf32>> -> tensor<131072xf32>
    %7 = tensor.empty() : tensor<131072x128xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %5, %6 : tensor<131072x128xi4>, tensor<131072xf32>, tensor<131072xf32>) outs(%7 : tensor<131072x128xf32>) {
    ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
      %9 = arith.extui %in : i4 to i32
      %10 = arith.uitofp %9 : i32 to f32
      %11 = arith.subf %10, %in_1 : f32
      %12 = arith.mulf %11, %in_0 : f32
      linalg.yield %12 : f32
    } -> tensor<131072x128xf32>
    flow.dispatch.tensor.store %8, %3, offsets = [0, 0], sizes = [131072, 128], strides = [1, 1] : tensor<131072x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<131072x128xf32>>
    return
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[8, 128], [2, 8]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [16, 4, 1]>
//       CHECK: func.func @i4_dequant()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

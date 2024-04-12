// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target))" %s | FileCheck %s

#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>
#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @forward_dispatch_0_generic_320x320x3x3() attributes {hal.executable.target = #executable_target_cuda_nvptx_fb} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x320x320x3xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<320x320x3x3xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [3, 320, 320, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x320x320x3xf32>> -> tensor<3x320x320x3xf32>
    %3 = tensor.empty() : tensor<320x320x3x3xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<3x320x320x3xf32>) outs(%3 : tensor<320x320x3x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %cst : f32
      linalg.yield %5 : f32
    } -> tensor<320x320x3x3xf32>
    flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [320, 320, 3, 3], strides = [1, 1, 1, 1] : tensor<320x320x3x3xf32> -> !flow.dispatch.tensor<writeonly:tensor<320x320x3x3xf32>>
    return
  }
}
//      CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUVectorize workgroup_size = [3, 3, 7] subgroup_size = 32>
//      CHECK: func.func @forward_dispatch_0_generic_320x320x3x3()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]

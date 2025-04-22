// RUN: iree-opt --split-input-file --iree-gpu-test-target=sm_60 --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @forward_dispatch_0_generic_320x320x3x3() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x320x320x3xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<320x320x3x3xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [3, 320, 320, 3], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x320x320x3xf32>> -> tensor<3x320x320x3xf32>
  %3 = tensor.empty() : tensor<320x320x3x3xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%2 : tensor<3x320x320x3xf32>) outs(%3 : tensor<320x320x3x3xf32>) {
  ^bb0(%in: f32, %out: f32):
    %5 = arith.addf %in, %cst : f32
    linalg.yield %5 : f32
  } -> tensor<320x320x3x3xf32>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [320, 320, 3, 3], strides = [1, 1, 1, 1] : tensor<320x320x3x3xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<320x320x3x3xf32>>
  return
}
//      CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32>
//      CHECK: func.func @forward_dispatch_0_generic_320x320x3x3()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]

// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0)>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @outer_reduction() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x16384xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>
  %in = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 16384], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x16384xf32>> -> tensor<512x16384xf32>
  %init = tensor.empty() : tensor<16384xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<16384xf32>) -> tensor<16384xf32>
  %result = linalg.generic {
      indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]}
      ins(%in : tensor<512x16384xf32>) outs(%fill : tensor<16384xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      %add = arith.addf %arg0, %arg1 : f32
      linalg.yield %add : f32
  } -> tensor<16384xf32>
  iree_tensor_ext.dispatch.tensor.store %result, %1, offsets = [0], sizes = [16384], strides = [1]
      : tensor<16384xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>
  return
}
// CHECK-LABEL: func.func @outer_reduction
//  CHECK-SAME:    #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [64, 1, 1] subgroup_size = 64
//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     reduction = [0, 4]
//  CHECK-SAME:     thread = [4, 0]
//  CHECK-SAME:     workgroup = [256, 0]

// -----

#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
#pipeline_layout_inner = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @inner_reduction() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_inner) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384x512xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_inner) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>
  %in = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16384, 512], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384x512xf32>> -> tensor<16384x512xf32>
  %init = tensor.empty() : tensor<16384xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<16384xf32>) -> tensor<16384xf32>
  %result = linalg.generic {
      indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction"]}
      ins(%in : tensor<16384x512xf32>) outs(%fill : tensor<16384xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      %add = arith.addf %arg0, %arg1 : f32
      linalg.yield %add : f32
  } -> tensor<16384xf32>
  iree_tensor_ext.dispatch.tensor.store %result, %1, offsets = [0], sizes = [16384], strides = [1]
      : tensor<16384xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>
  return
}
// CHECK-LABEL: func.func @inner_reduction
//   CHECK-NOT: TileAndFuse

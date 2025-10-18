// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --iree-config-add-tuner-attributes --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' %s | FileCheck %s

func.func @matmul(%lhs: tensor<4x4xf32>, %rhs: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<4x4xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<4x4xf32>, tensor<4x4xf32>)
             outs(%fill: tensor<4x4xf32>) -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}

// CHECK: %2 = linalg.matmul {lowering_config = #{{.*}}, root_op} ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%1 : tensor<4x4xf32>) -> tensor<4x4xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @matvec_like() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x32000xf16>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xf16>> -> tensor<1x4096xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
  %5 = tensor.empty() : tensor<1x32000xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<1x32000xf16>) -> tensor<1x32000xf16>
  %7 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%3, %4 : tensor<1x4096xf16>, tensor<32000x4096xf16>)
    outs(%6 : tensor<1x32000xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %8 = arith.mulf %in, %in_0 : f16
    %9 = arith.addf %out, %8 : f16
    linalg.yield %9 : f16
  } -> tensor<1x32000xf16>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 32000], strides = [1, 1] : tensor<1x32000xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x32000xf16>>
  return
}

// CHECK: #translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
// CHECK-LABEL: func.func @matvec_like
// CHECK: %{{.*}} = linalg.generic
// CHECK-SAME: lowering_config = #iree_gpu.lowering_config
// CHECK-SAME: root_op

// -----

#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#pipeline_layout2 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @reduction_sum() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout2) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x32x128x4096xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout2) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 32, 128, 4096], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x32x128x4096xf32>> -> tensor<2x32x128x4096xf32>
  %3 = tensor.empty() : tensor<2x32xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2x32xf32>) -> tensor<2x32xf32>
  %5 = linalg.generic {
    indexing_maps = [#map3, #map4],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } ins(%2 : tensor<2x32x128x4096xf32>)
    outs(%4 : tensor<2x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<2x32xf32>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0, 0], sizes = [2, 32], strides = [1, 1] : tensor<2x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32xf32>>
  return
}

// CHECK: #translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
// CHECK-LABEL: func.func @reduction_sum
// CHECK: %{{.*}} = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME: lowering_config = #iree_gpu.lowering_config
// CHECK-SAME: root_op

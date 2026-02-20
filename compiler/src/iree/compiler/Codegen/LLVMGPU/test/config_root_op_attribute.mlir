// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --iree-codegen-add-tuner-attributes --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' %s | FileCheck %s

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

func.func @matvec(%matrix: tensor<32000x4096xf16>, %vector: tensor<4096xf16>, %init: tensor<32000xf16>) {
  %output = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>]>) binding(0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32000xf16>>
  %result = linalg.matvec ins(%matrix, %vector : tensor<32000x4096xf16>, tensor<4096xf16>) outs(%init : tensor<32000xf16>) -> tensor<32000xf16>
  iree_tensor_ext.dispatch.tensor.store %result, %output, offsets = [0], sizes = [32000], strides = [1] : tensor<32000xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32000xf16>>
  return
}

// CHECK: #translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
// CHECK-LABEL: func.func @matvec
// CHECK: linalg.matvec
// CHECK-SAME: lowering_config = #iree_gpu.lowering_config
// CHECK-SAME: root_op

// -----

#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

func.func @reduction_sum(%input: tensor<2x32x128x4096xf32>, %init: tensor<2x32xf32>) {
  %output = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>]>) binding(0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32xf32>>
  %result = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
      ins(%input : tensor<2x32x128x4096xf32>) outs(%init : tensor<2x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %add = arith.addf %in, %out : f32
    linalg.yield %add : f32
  } -> tensor<2x32xf32>
  iree_tensor_ext.dispatch.tensor.store %result, %output, offsets = [0, 0], sizes = [2, 32], strides = [1, 1] : tensor<2x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32xf32>>
  return
}

// CHECK: #translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
// CHECK-LABEL: func.func @reduction_sum
// CHECK: %{{.*}} = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME: lowering_config = #iree_gpu.lowering_config
// CHECK-SAME: root_op

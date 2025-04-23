// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-drop-resource-casts))" | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>
func.func @simple_cast() -> (tensor<2xf32>, tensor<2xf32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %arg0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>>
  %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets=[0], sizes=[2], strides=[1]
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
  %1 = iree_gpu.buffer_resource_cast %0 : tensor<2xf32>
  return %0, %1 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @simple_cast
//       CHECK:   %[[LOAD:.+]] = iree_tensor_ext.dispatch.tensor.load
//       CHECK:   %[[CAST:.+]] = iree_gpu.buffer_resource_cast
//       CHECK:   return %[[LOAD]], %[[CAST]] : tensor<2xf32>, tensor<2xf32>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>
func.func @drop_multiuse_producer_swizzle_cast() -> (tensor<2xf32>, tensor<2xf32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %arg0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>>
  %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets=[0], sizes=[2], strides=[1]
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
  %1 = iree_gpu.buffer_resource_cast %0 cacheSwizzleStride(%c2) : tensor<2xf32>
  return %0, %1 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @drop_multiuse_producer_swizzle_cast
//       CHECK:   %[[LOAD:.+]] = iree_tensor_ext.dispatch.tensor.load
//       CHECK:   return %[[LOAD]], %[[LOAD]] : tensor<2xf32>, tensor<2xf32>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>
func.func @drop_binding_annotation() -> tensor<2xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %arg0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
    {iree_gpu.use_rocdl_buffer_instructions}
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>>
  %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets=[0], sizes=[2], strides=[1]
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
  %1 = iree_gpu.buffer_resource_cast %0 cacheSwizzleStride(%c2) : tensor<2xf32>
  return %1 : tensor<2xf32>
}

// CHECK-LABEL: func.func @drop_binding_annotation
//       CHECK:   hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%{{.*}}) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>>
//       CHECK:   %[[CAST:.+]] = iree_gpu.buffer_resource_cast
//       CHECK:   return %[[CAST]] : tensor<2xf32>

// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-bubble-resource-casts))" | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>
func.func @simple_cast() -> (tensor<2xf32>, tensor<2xf32>) {
  %c0 = arith.constant 0 : index
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

// -----

func.func @bubble_extract_slice(%arg0: tensor<4xf32>) -> tensor<2xf32> {
  %0 = tensor.extract_slice %arg0 [0] [2] [1] : tensor<4xf32> to tensor<2xf32>
  %1 = iree_gpu.buffer_resource_cast %0 : tensor<2xf32>
  return %1 : tensor<2xf32>
}

// CHECK-LABEL: func.func @bubble_extract_slice
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<4xf32>
//       CHECK:   %[[CAST:.+]] = iree_gpu.buffer_resource_cast %[[ARG0]]
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[CAST]]
//       CHECK:   return %[[EXTRACT]]

// -----

func.func @bubble_pad(%arg0: tensor<1xf32>) -> tensor<2xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.pad %arg0 low[0] high[1] {
  ^bb0(%arg1: index):
    tensor.yield %cst : f32
  } : tensor<1xf32> to tensor<2xf32>
  %1 = iree_gpu.buffer_resource_cast %0 : tensor<2xf32>
  return %1 : tensor<2xf32>
}

// CHECK-LABEL: func.func @bubble_pad
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<1xf32>
//       CHECK:   %[[CAST:.+]] = iree_gpu.buffer_resource_cast %[[ARG0]]
//       CHECK:   %[[PAD:.+]] = tensor.pad %[[CAST]]
//       CHECK:   return %[[PAD]]

// -----

func.func @bubble_expand_and_collapse(%arg0: tensor<3x2xf32>) -> tensor<2x3xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<3x2xf32> into tensor<6xf32>
  %1 = tensor.expand_shape %0 [[0, 1]] output_shape [3, 2]
      : tensor<6xf32> into tensor<2x3xf32>
  %2 = iree_gpu.buffer_resource_cast %1 : tensor<2x3xf32>
  return %2 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @bubble_expand_and_collapse
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<3x2xf32>
//       CHECK:   %[[CAST:.+]] = iree_gpu.buffer_resource_cast %[[ARG0]]
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[CAST]]
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[COLLAPSE]]
//       CHECK:   return %[[EXPAND]]

// -----

func.func @drop_blocked_resource_cast(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.pad %arg0 low[0] high[1] {
  ^bb0(%arg1: index):
    tensor.yield %cst : f32
  } : tensor<?xf32> to tensor<?xf32>
  %swizzle = tensor.dim %arg0, %c0 : tensor<?xf32>
  %1 = iree_gpu.buffer_resource_cast %0 cacheSwizzleStride(%swizzle) : tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: func.func @drop_blocked_resource_cast
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<?xf32>
//       CHECK:   %[[PAD:.+]] = tensor.pad %[[ARG0]]
//       CHECK:   return %[[PAD]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>
func.func @keep_swizzle_cast() -> tensor<1xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %arg0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>>
  %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets=[0], sizes=[2], strides=[1]
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
  %1 = tensor.extract_slice %0 [0] [1] [1] : tensor<2xf32> to tensor<1xf32>
  %2 = iree_gpu.buffer_resource_cast %1 cacheSwizzleStride(%c2) : tensor<1xf32>
  return %2 : tensor<1xf32>
}

// CHECK-LABEL: func.func @keep_swizzle_cast
//       CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan
//       CHECK:   %[[LOAD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[BINDING]]
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[LOAD]]
//       CHECK:   %[[CAST:.+]] = iree_gpu.buffer_resource_cast %[[EXTRACT]]
//       CHECK:   return %[[CAST]] : tensor<1xf32>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>
func.func @bufferized_subspan_drop_buffer_cast() -> (tensor<2xf32>, tensor<2xf32>) {
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly") : memref<2xf32, #hal.descriptor_type<storage_buffer>>
  %1 = amdgpu.fat_raw_buffer_cast %0 : memref<2xf32, #hal.descriptor_type<storage_buffer>> to memref<2xf32, #amdgpu.address_space<fat_raw_buffer>>
  %2 = iree_codegen.load_from_buffer %1 : memref<2xf32, #amdgpu.address_space<fat_raw_buffer>> -> tensor<2xf32>
  %3 = iree_gpu.buffer_resource_cast %2 cacheSwizzleStride(%c4096) : tensor<2xf32>
  return %2, %3 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @bufferized_subspan_drop_buffer_cast
//       CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan
//   CHECK-NOT:   amdgpu.fat_raw_buffer_cast
//       CHECK:   %[[LOAD:.+]] = iree_codegen.load_from_buffer %[[BINDING]]
//       CHECK:   %[[CAST:.+]] = iree_gpu.buffer_resource_cast %[[LOAD]]
//       CHECK:   return %[[LOAD]], %[[CAST]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>
func.func @bufferized_subspan_multiple_buffer_casts() -> (tensor<2xf32>, tensor<2xf32>) {
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly") : memref<2xf32, #hal.descriptor_type<storage_buffer>>
  %1 = amdgpu.fat_raw_buffer_cast %0 : memref<2xf32, #hal.descriptor_type<storage_buffer>> to memref<2xf32, #amdgpu.address_space<fat_raw_buffer>>
  %2 = amdgpu.fat_raw_buffer_cast %0 : memref<2xf32, #hal.descriptor_type<storage_buffer>> to memref<2xf32, #amdgpu.address_space<fat_raw_buffer>>
  %3 = iree_codegen.load_from_buffer %1 : memref<2xf32, #amdgpu.address_space<fat_raw_buffer>> -> tensor<2xf32>
  %4 = iree_codegen.load_from_buffer %2 : memref<2xf32, #amdgpu.address_space<fat_raw_buffer>> -> tensor<2xf32>
  %5 = iree_gpu.buffer_resource_cast %4 cacheSwizzleStride(%c4096) : tensor<2xf32>
  return %3, %5 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @bufferized_subspan_multiple_buffer_casts
//       CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan
//       CHECK:   %[[CAST0:.+]] = amdgpu.fat_raw_buffer_cast %[[BINDING]]
//       CHECK:   %[[CAST1:.+]] = amdgpu.fat_raw_buffer_cast %[[BINDING]]
//       CHECK:   %[[LOAD0:.+]] = iree_codegen.load_from_buffer %[[CAST0]]
//       CHECK:   %[[LOAD1:.+]] = iree_codegen.load_from_buffer %[[CAST1]]
//       CHECK:   return %[[LOAD0]], %[[LOAD1]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>
func.func @bufferized_subspan_not_readonly() -> (tensor<2xf32>, tensor<2xf32>) {
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("Indirect") : memref<2xf32, #hal.descriptor_type<storage_buffer>>
  %1 = amdgpu.fat_raw_buffer_cast %0 : memref<2xf32, #hal.descriptor_type<storage_buffer>> to memref<2xf32, #amdgpu.address_space<fat_raw_buffer>>
  %2 = iree_codegen.load_from_buffer %1 : memref<2xf32, #amdgpu.address_space<fat_raw_buffer>> -> tensor<2xf32>
  %3 = iree_gpu.buffer_resource_cast %2 cacheSwizzleStride(%c4096) : tensor<2xf32>
  return %2, %3 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @bufferized_subspan_not_readonly
//       CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan
//       CHECK:   %[[BUFFER_CAST:.+]] = amdgpu.fat_raw_buffer_cast %[[BINDING]]
//       CHECK:   %[[LOAD:.+]] = iree_codegen.load_from_buffer %[[BUFFER_CAST]]
//       CHECK:   return %[[LOAD]], %[[LOAD]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>
func.func @bufferized_subspan_no_buffer_cast() -> (tensor<2xf32>, tensor<2xf32>) {
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly") : memref<2xf32, #hal.descriptor_type<storage_buffer>>
  %1 = iree_codegen.load_from_buffer %0 : memref<2xf32, #hal.descriptor_type<storage_buffer>> -> tensor<2xf32>
  %2 = iree_gpu.buffer_resource_cast %1 cacheSwizzleStride(%c4096) : tensor<2xf32>
  return %1, %2 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @bufferized_subspan_no_buffer_cast
//       CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan
//       CHECK:   %[[LOAD:.+]] = iree_codegen.load_from_buffer %[[BINDING]]
//       CHECK:   %[[CAST:.+]] = iree_gpu.buffer_resource_cast %[[LOAD]]
//       CHECK:   return %[[LOAD]], %[[CAST]]

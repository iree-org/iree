// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-bufferize-dispatch-tensor-load-store,cse))" -split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @dispatch_tensor_load_and_store() {
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [16], strides = [1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<16xf32>
  iree_tensor_ext.dispatch.tensor.store %3, %2, offsets = [0], sizes = [16], strides = [1]
      : tensor<16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  return
}

// CHECK-LABEL: func.func @dispatch_tensor_load_and_store()
// CHECK:         %[[INPUT:.+]] = hal.interface.binding.subspan
// CHECK-SAME:        binding(0) : memref<16xf32, #hal.descriptor_type<storage_buffer>>
// CHECK:         %[[OUTPUT:.+]] = hal.interface.binding.subspan
// CHECK-SAME:        binding(1) : memref<16xf32, #hal.descriptor_type<storage_buffer>>
// CHECK:         %[[LOAD:.+]] = iree_codegen.load_from_buffer %[[INPUT]]
// CHECK-SAME:        : memref<16xf32, #hal.descriptor_type<storage_buffer>> -> tensor<16xf32>
// CHECK:         iree_codegen.store_to_buffer %[[LOAD]], %[[OUTPUT]]
// CHECK-SAME:        : tensor<16xf32> into memref<16xf32, #hal.descriptor_type<storage_buffer>>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @dispatch_tensor_load_and_store_slices() {
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [2], sizes = [12], strides = [1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<12xf32>
  iree_tensor_ext.dispatch.tensor.store %3, %2, offsets = [4], sizes = [12], strides = [1]
      : tensor<12xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  return
}

// CHECK-LABEL: func.func @dispatch_tensor_load_and_store_slices()
// CHECK:         %[[INPUT:.+]] = hal.interface.binding.subspan
// CHECK-SAME:        binding(0) : memref<16xf32, #hal.descriptor_type<storage_buffer>>
// CHECK:         %[[OUTPUT:.+]] = hal.interface.binding.subspan
// CHECK-SAME:        binding(1) : memref<16xf32, #hal.descriptor_type<storage_buffer>>
// CHECK:         %[[OUTPUT_SUBVIEW:.+]] = memref.subview %[[OUTPUT]][4] [12] [1]
// CHECK-SAME:        : memref<16xf32, #hal.descriptor_type<storage_buffer>> to
// CHECK-SAME:          memref<12xf32, strided<[1], offset: 4>, #hal.descriptor_type<storage_buffer>>
// CHECK:         %[[INPUT_SUBVIEW:.+]] = memref.subview %[[INPUT]][2] [12] [1]
// CHECK-SAME:        : memref<16xf32, #hal.descriptor_type<storage_buffer>> to
// CHECK-SAME:          memref<12xf32, strided<[1], offset: 2>, #hal.descriptor_type<storage_buffer>>
// CHECK:         %[[LOAD:.+]] = iree_codegen.load_from_buffer %[[INPUT_SUBVIEW]]
// CHECK-SAME:        : memref<12xf32, strided<[1], offset: 2>, #hal.descriptor_type<storage_buffer>> -> tensor<12xf32>
// CHECK:         iree_codegen.store_to_buffer %[[LOAD]], %[[OUTPUT_SUBVIEW]]
// CHECK-SAME:        : tensor<12xf32> into memref<12xf32, strided<[1], offset: 4>, #hal.descriptor_type<storage_buffer>>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @dispatch_tensor_load_and_store_with_compute_op() {
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [2], sizes = [12], strides = [1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<12xf32>
  %init = tensor.empty() : tensor<12xf32>
  %copy = linalg.copy ins(%3 : tensor<12xf32>) outs(%init : tensor<12xf32>) -> tensor<12xf32>
  iree_tensor_ext.dispatch.tensor.store %copy, %2, offsets = [4], sizes = [12], strides = [1]
      : tensor<12xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
  return
}

// CHECK-LABEL: func.func @dispatch_tensor_load_and_store_with_compute_op()
// CHECK:         %[[INPUT:.+]] = hal.interface.binding.subspan
// CHECK-SAME:        binding(0) : memref<16xf32, #hal.descriptor_type<storage_buffer>>
// CHECK:         %[[OUTPUT:.+]] = hal.interface.binding.subspan
// CHECK-SAME:        binding(1) : memref<16xf32, #hal.descriptor_type<storage_buffer>>
// CHECK:         %[[OUTPUT_SUBVIEW:.+]] = memref.subview %[[OUTPUT]][4] [12] [1]
// CHECK:         %[[INPUT_SUBVIEW:.+]] = memref.subview %[[INPUT]][2] [12] [1]
// CHECK:         %[[LOAD:.+]] = iree_codegen.load_from_buffer %[[INPUT_SUBVIEW]]
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<12xf32>
// CHECK:         %[[COPY:.+]] = linalg.copy ins(%[[LOAD]]{{.*}} outs(%[[INIT]]
// CHECK:         iree_codegen.store_to_buffer %[[COPY]], %[[OUTPUT_SUBVIEW]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @dynamic_dispatch_tensor_load_and_store(%offset: index, %size: index, %stride: index, %binding_size: index) {
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%binding_size}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%binding_size}
  %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [%offset], sizes = [%size], strides = [%stride]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%binding_size} -> tensor<?xf32>
  iree_tensor_ext.dispatch.tensor.store %3, %2, offsets = [%offset], sizes = [%size], strides = [%stride]
      : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%binding_size}
  return
}

// CHECK-LABEL: func.func @dynamic_dispatch_tensor_load_and_store
// CHECK-SAME:    %[[OFFSET:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[SIZE:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[STRIDE:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[BINDING_SIZE:[a-zA-Z0-9_]+]]
// CHECK:         %[[INPUT:.+]] = hal.interface.binding.subspan
// CHECK-SAME:        binding(0) : memref<?xf32, #hal.descriptor_type<storage_buffer>>{%[[BINDING_SIZE]]}
// CHECK:         %[[OUTPUT:.+]] = hal.interface.binding.subspan
// CHECK-SAME:        binding(1) : memref<?xf32, #hal.descriptor_type<storage_buffer>>{%[[BINDING_SIZE]]}
// CHECK:         %[[OUTPUT_SUBVIEW:.+]] = memref.subview %[[OUTPUT]][%[[OFFSET]]] [%[[SIZE]]] [%[[STRIDE]]]
// CHECK-SAME:        : memref<?xf32, #hal.descriptor_type<storage_buffer>> to
// CHECK-SAME:          memref<?xf32, strided<[?], offset: ?>, #hal.descriptor_type<storage_buffer>>
// CHECK:         %[[INPUT_SUBVIEW:.+]] = memref.subview %[[INPUT]][%[[OFFSET]]] [%[[SIZE]]] [%[[STRIDE]]]
// CHECK-SAME:        : memref<?xf32, #hal.descriptor_type<storage_buffer>> to
// CHECK-SAME:          memref<?xf32, strided<[?], offset: ?>, #hal.descriptor_type<storage_buffer>>
// CHECK:         %[[LOAD:.+]] = iree_codegen.load_from_buffer %[[INPUT_SUBVIEW]]
// CHECK-SAME:        : memref<?xf32, strided<[?], offset: ?>, #hal.descriptor_type<storage_buffer>> -> tensor<?xf32>
// CHECK:         iree_codegen.store_to_buffer %[[LOAD]], %[[OUTPUT_SUBVIEW]]
// CHECK-SAME:        : tensor<?xf32> into memref<?xf32, strided<[?], offset: ?>, #hal.descriptor_type<storage_buffer>>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @rank_reducing_slices() {
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x16xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x16xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 2], sizes = [1, 12], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x16xf32>> -> tensor<12xf32>
  iree_tensor_ext.dispatch.tensor.store %3, %2, offsets = [0, 4], sizes = [1, 12], strides = [1, 1]
      : tensor<12xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x16xf32>>
  return
}

// CHECK-LABEL: func.func @rank_reducing_slices()
// CHECK:         %[[INPUT:.+]] = hal.interface.binding.subspan
// CHECK-SAME:        binding(0) : memref<8x16xf32, #hal.descriptor_type<storage_buffer>>
// CHECK:         %[[OUTPUT:.+]] = hal.interface.binding.subspan
// CHECK-SAME:        binding(1) : memref<8x16xf32, #hal.descriptor_type<storage_buffer>>
// CHECK:         %[[OUTPUT_SUBVIEW:.+]] = memref.subview %[[OUTPUT]][0, 4] [1, 12] [1, 1]
// CHECK-SAME:        : memref<8x16xf32, #hal.descriptor_type<storage_buffer>> to
// CHECK-SAME:          memref<12xf32, strided<[1], offset: 4>, #hal.descriptor_type<storage_buffer>>
// CHECK:         %[[INPUT_SUBVIEW:.+]] = memref.subview %[[INPUT]][0, 2] [1, 12] [1, 1]
// CHECK-SAME:        : memref<8x16xf32, #hal.descriptor_type<storage_buffer>> to
// CHECK-SAME:          memref<12xf32, strided<[1], offset: 2>, #hal.descriptor_type<storage_buffer>>
// CHECK:         %[[LOAD:.+]] = iree_codegen.load_from_buffer %[[INPUT_SUBVIEW]]
// CHECK-SAME:        : memref<12xf32, strided<[1], offset: 2>, #hal.descriptor_type<storage_buffer>> -> tensor<12xf32>
// CHECK:         iree_codegen.store_to_buffer %[[LOAD]], %[[OUTPUT_SUBVIEW]]
// CHECK-SAME:        : tensor<12xf32> into memref<12xf32, strided<[1], offset: 4>, #hal.descriptor_type<storage_buffer>>

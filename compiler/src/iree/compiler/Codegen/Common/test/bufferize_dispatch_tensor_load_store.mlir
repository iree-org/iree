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

// -----

// Verifies that dispatch.tensor.load ops nested inside scf.forall (e.g. from
// split reduction) are bufferized correctly with memref.subview.

#pipeline_layout_nested = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @nested_dispatch_tensor_load_in_forall() {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_nested) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24576x512xbf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_nested) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3072x512x8xbf16>>
  %empty = tensor.empty() : tensor<3072x512x8xbf16>
  %result = scf.forall (%iv) = (0) to (24576) step (3072) shared_outs(%out = %empty) -> (tensor<3072x512x8xbf16>) {
    %load = iree_tensor_ext.dispatch.tensor.load %0, offsets = [%iv, 0], sizes = [3072, 512], strides = [1, 1]
        : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24576x512xbf16>> -> tensor<3072x512xbf16>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %load into %out[0, 0, 0] [3072, 512, 1] [1, 1, 1] : tensor<3072x512xbf16> into tensor<3072x512x8xbf16>
    }
  }
  iree_tensor_ext.dispatch.tensor.store %result, %1, offsets = [0, 0, 0], sizes = [3072, 512, 8], strides = [1, 1, 1]
      : tensor<3072x512x8xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3072x512x8xbf16>>
  return
}

// CHECK-LABEL: func.func @nested_dispatch_tensor_load_in_forall()
// CHECK-DAG:     %[[INPUT:.+]] = hal.interface.binding.subspan {{.+}} binding(0) : memref<24576x512xbf16, #hal.descriptor_type<storage_buffer>>
// CHECK-DAG:     %[[OUTPUT:.+]] = hal.interface.binding.subspan {{.+}} binding(1) : memref<3072x512x8xbf16, #hal.descriptor_type<storage_buffer>>
// CHECK:         scf.forall (%[[IV:.+]]) =
// CHECK:           %[[SUBVIEW:.+]] = memref.subview %[[INPUT]][%[[IV]], 0] [3072, 512] [1, 1]
// CHECK-SAME:          : memref<24576x512xbf16, #hal.descriptor_type<storage_buffer>> to
// CHECK-SAME:            memref<3072x512xbf16, strided<[512, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
// CHECK:           iree_codegen.load_from_buffer %[[SUBVIEW]]
// CHECK-SAME:          : memref<3072x512xbf16, strided<[512, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> -> tensor<3072x512xbf16>
// CHECK:         iree_codegen.store_to_buffer %{{.+}}, %[[OUTPUT]]

// -----

// Verify that dispatch.tensor.load with negative strides is correctly
// bufferized to a memref.subview with negative strides, not the raw buffer.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @negative_stride_load_store() {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [3], sizes = [4], strides = [-1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<4xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %1, offsets = [0], sizes = [4], strides = [1]
      : tensor<4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>
  return
}

// CHECK-LABEL: func.func @negative_stride_load_store()
// CHECK:         %[[INPUT:.+]] = hal.interface.binding.subspan
// CHECK-SAME:        binding(0) : memref<4xf32, #hal.descriptor_type<storage_buffer>>
// CHECK:         %[[OUTPUT:.+]] = hal.interface.binding.subspan
// CHECK-SAME:        binding(1) : memref<4xf32, #hal.descriptor_type<storage_buffer>>
// CHECK:         %[[SUBVIEW:.+]] = memref.subview %[[INPUT]][3] [4] [-1]
// CHECK-SAME:        : memref<4xf32, #hal.descriptor_type<storage_buffer>> to
// CHECK-SAME:          memref<4xf32, strided<[-1], offset: 3>, #hal.descriptor_type<storage_buffer>>
// CHECK:         %[[LOAD:.+]] = iree_codegen.load_from_buffer %[[SUBVIEW]]
// CHECK-SAME:        : memref<4xf32, strided<[-1], offset: 3>, #hal.descriptor_type<storage_buffer>> -> tensor<4xf32>
// CHECK:         iree_codegen.store_to_buffer %[[LOAD]], %[[OUTPUT]]
// CHECK-SAME:        : tensor<4xf32> into memref<4xf32, #hal.descriptor_type<storage_buffer>>

// RUN: iree-opt -allow-unregistered-dialect -split-input-file %s | iree-opt -allow-unregistered-dialect -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @buffer_view_create
func @buffer_view_create(%arg0: !hal.buffer) -> !hal.buffer_view {
  %c32 = constant 32 : i32
  %0:2 = "test_hal.shape"() : () -> (index, index)
  // CHECK: %view = hal.buffer_view.create %arg0, element_type = %c32_i32, shape = [%0#0, %0#1] : !hal.buffer -> !hal.buffer_view
  %view = hal.buffer_view.create %arg0, element_type = %c32, shape = [%0#0, %0#1] : !hal.buffer -> !hal.buffer_view
  return %view : !hal.buffer_view
}

// -----

// CHECK-LABEL: @buffer_view_buffer
func @buffer_view_buffer(%arg0: !hal.buffer_view) -> !hal.buffer {
  // CHECK: %buffer = hal.buffer_view.buffer %arg0 : !hal.buffer
  %buffer = hal.buffer_view.buffer %arg0 : !hal.buffer
  return %buffer : !hal.buffer
}

// -----

// CHECK-LABEL: @buffer_view_byte_length
func @buffer_view_byte_length(%arg0: !hal.buffer_view) -> index {
  // CHECK: %len = hal.buffer_view.byte_length %arg0 : index
  %len = hal.buffer_view.byte_length %arg0 : index
  return %len : index
}

// -----

// CHECK-LABEL: @buffer_view_shape_queries
func @buffer_view_shape_queries(%arg0: !hal.buffer_view) -> (index, index, index, index) {
  // CHECK: %{{.+}} = hal.buffer_view.rank %arg0 : index
  %0 = hal.buffer_view.rank %arg0 : index
  // CHECK: %{{.+}} = hal.buffer_view.dim %arg0, 0 : index
  %1 = hal.buffer_view.dim %arg0, 0 : index
  // CHECK: %{{.+}}:2 = hal.buffer_view.dims %arg0 : index, index
  %2, %3 = hal.buffer_view.dims %arg0 : index, index
  return %0, %1, %2, %3 : index, index, index, index
}

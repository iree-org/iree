// Tests printing and parsing of hal.buffer_view ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// -----

// CHECK-LABEL: @buffer_view_const
func @buffer_view_const() -> !iree.ref<!hal.buffer_view> {
  %0 = "test_hal.allocator"() : () -> !iree.ref<!hal.allocator>
  // CHECK: %view = hal.buffer_view.const %0, "HostVisible|HostCoherent", "Transfer" : !iree.ref<!hal.buffer_view> = dense<[4, 1, 2]> : tensor<3xi32>
  %view = hal.buffer_view.const %0, "HostVisible|HostCoherent", "Transfer" : !iree.ref<!hal.buffer_view> = dense<[4, 1, 2]> : tensor<3xi32>
  return %view : !iree.ref<!hal.buffer_view>
}

// -----

// CHECK-LABEL: @buffer_view_create
func @buffer_view_create(%arg0 : !iree.ref<!hal.buffer>) -> !iree.ref<!hal.buffer_view> {
  %0:2 = "test_hal.shape"() : () -> (i32, i32)
  // CHECK: %view = hal.buffer_view.create %arg0, shape=[%0#0, %0#1], element_type=32 : !iree.ref<!hal.buffer_view>
  %view = hal.buffer_view.create %arg0, shape=[%0#0, %0#1], element_type=32 : !iree.ref<!hal.buffer_view>
  return %view : !iree.ref<!hal.buffer_view>
}

// -----

// CHECK-LABEL: @buffer_view_subview
func @buffer_view_subview(%arg0 : !iree.ref<!hal.buffer_view>) -> !iree.ref<!hal.buffer_view> {
  %0:2 = "test_hal.indices"() : () -> (i32, i32)
  %1:2 = "test_hal.lengths"() : () -> (i32, i32)
  // CHECK: %view = hal.buffer_view.subview %arg0, indices = [%0#0, %0#1], lengths = [%1#0, %1#1] : !iree.ref<!hal.buffer_view>
  %view = hal.buffer_view.subview %arg0, indices = [%0#0, %0#1], lengths = [%1#0, %1#1] : !iree.ref<!hal.buffer_view>
  return %view : !iree.ref<!hal.buffer_view>
}

// -----

// CHECK-LABEL: @buffer_view_buffer
func @buffer_view_buffer(%arg0 : !iree.ref<!hal.buffer_view>) -> !iree.ref<!hal.buffer> {
  // CHECK: %buffer = hal.buffer_view.buffer %arg0 : !iree.ref<!hal.buffer>
  %buffer = hal.buffer_view.buffer %arg0 : !iree.ref<!hal.buffer>
  return %buffer : !iree.ref<!hal.buffer>
}

// -----

// CHECK-LABEL: @buffer_view_byte_length
func @buffer_view_byte_length(%arg0 : !iree.ref<!hal.buffer_view>) -> i32 {
  // CHECK: %len = hal.buffer_view.byte_length %arg0 : i32
  %len = hal.buffer_view.byte_length %arg0 : i32
  return %len : i32
}

// -----

// CHECK-LABEL: @buffer_view_compute_offset
func @buffer_view_compute_offset(%arg0 : !iree.ref<!hal.buffer_view>) -> i32 {
  %0:2 = "test_hal.indices"() : () -> (i32, i32)
  // CHECK: %off = hal.buffer_view.compute_offset %arg0, indices = [%0#0, %0#1]
  %off = hal.buffer_view.compute_offset %arg0, indices = [%0#0, %0#1]
  return %off : i32
}

// -----

// CHECK-LABEL: @buffer_view_compute_range
func @buffer_view_compute_range(%arg0 : !iree.ref<!hal.buffer_view>) -> (i32, i32) {
  %0:2 = "test_hal.indices"() : () -> (i32, i32)
  %1:2 = "test_hal.lengths"() : () -> (i32, i32)
  // CHECK: %off, %len = hal.buffer_view.compute_range %arg0, indices = [%0#0, %0#1], lengths = [%1#0, %1#1]
  %off, %len = hal.buffer_view.compute_range %arg0, indices = [%0#0, %0#1], lengths = [%1#0, %1#1]
  return %off, %len : i32, i32
}

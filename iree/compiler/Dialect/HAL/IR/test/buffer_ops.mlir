// Tests printing and parsing of hal.buffer ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @buffer_subspan
func @buffer_subspan() -> !iree.ref<!hal.buffer> {
  %0 = "test_hal.buffer"() : () -> !iree.ref<!hal.buffer>
  %1 = "test_hal.device_size"() : () -> i32
  %2 = "test_hal.device_size"() : () -> i32
  // CHECK: %buffer = hal.buffer.subspan %0, %1, %2 : !iree.ref<!hal.buffer>
  %buffer = hal.buffer.subspan %0, %1, %2 : !iree.ref<!hal.buffer>
  return %buffer : !iree.ref<!hal.buffer>
}

// -----

// CHECK-LABEL: @buffer_fill
func @buffer_fill(%arg0 : !iree.ref<!hal.buffer>) {
  %0 = "test_hal.device_size"() : () -> i32
  %1 = "test_hal.device_size"() : () -> i32
  %2 = "test_hal.pattern"() : () -> i32
  // CHECK: hal.buffer.fill %arg0, %0, %1, %2
  hal.buffer.fill %arg0, %0, %1, %2
  return
}

// -----

// CHECK-LABEL: @buffer_read_data
func @buffer_read_data(%arg0 : !iree.ref<!hal.buffer>) {
  %0 = "test_hal.device_size"() : () -> i32
  %1 = "test_hal.mutable_data"() : () -> !iree.mutable_byte_buffer_ref
  %2 = "test_hal.device_size"() : () -> i32
  %3 = "test_hal.device_size"() : () -> i32
  // CHECK: hal.buffer.read_data %arg0, %0, %1, %2, %3 : !iree.mutable_byte_buffer_ref
  hal.buffer.read_data %arg0, %0, %1, %2, %3 : !iree.mutable_byte_buffer_ref
  return
}

// -----

// CHECK-LABEL: @buffer_write_data
func @buffer_write_data(%arg0 : !iree.ref<!hal.buffer>) {
  %0 = "test_hal.mutable_data"() : () -> !iree.mutable_byte_buffer_ref
  %1 = "test_hal.device_size"() : () -> i32
  %2 = "test_hal.device_size"() : () -> i32
  %3 = "test_hal.device_size"() : () -> i32
  // CHECK: hal.buffer.write_data %0, %1, %arg0, %2, %3 : !iree.mutable_byte_buffer_ref
  hal.buffer.write_data %0, %1, %arg0, %2, %3 : !iree.mutable_byte_buffer_ref
  return
}

// -----

// CHECK-LABEL: @buffer_copy_data
func @buffer_copy_data(%arg0 : !iree.ref<!hal.buffer>, %arg1 : !iree.ref<!hal.buffer>) {
  %0 = "test_hal.device_size"() : () -> i32
  %1 = "test_hal.device_size"() : () -> i32
  %2 = "test_hal.device_size"() : () -> i32
  // CHECK: hal.buffer.copy_data %arg0, %0, %arg1, %1, %2
  hal.buffer.copy_data %arg0, %0, %arg1, %1, %2
  return
}

// -----

// CHECK-LABEL: @buffer_load
func @buffer_load(%arg0 : !iree.ref<!hal.buffer>) -> i32 {
  %0 = "test_hal.device_size"() : () -> i32
  // CHECK: [[VAL:%.+]] = hal.buffer.load %arg0[%0] : i32
  %1 = hal.buffer.load %arg0[%0] : i32
  // CHECK-NEXT: return [[VAL]]
  return %1 : i32
}

// -----

// CHECK-LABEL: @buffer_store
func @buffer_store(%arg0 : i32, %arg1 : !iree.ref<!hal.buffer>) {
  %0 = "test_hal.device_size"() : () -> i32
  // CHECK: hal.buffer.store %arg0, %arg1[%0] : i32
  hal.buffer.store %arg0, %arg1[%0] : i32
  return
}

// -----

// CHECK-LABEL: @buffer_view_compute_offset
func @buffer_view_compute_offset(%arg0 : !iree.ref<!hal.buffer>) -> i32 {
  %0:2 = "test_hal.shape"() : () -> (i32, i32)
  %1:2 = "test_hal.indices"() : () -> (i32, i32)
  // CHECK: %off = hal.buffer_view.compute_offset %arg0, shape=[%0#0, %0#1], indices=[%1#0, %1#1], element_size=4
  %off = hal.buffer_view.compute_offset %arg0, shape=[%0#0, %0#1], indices=[%1#0, %1#1], element_size=4
  return %off : i32
}

// -----

// CHECK-LABEL: @buffer_view_compute_length
func @buffer_view_compute_length(%arg0 : !iree.ref<!hal.buffer>) -> i32 {
  %0:2 = "test_hal.shape"() : () -> (i32, i32)
  // CHECK: %len = hal.buffer_view.compute_length %arg0, shape=[%0#0, %0#1], element_size=4
  %len = hal.buffer_view.compute_length %arg0, shape=[%0#0, %0#1], element_size=4
  return %len : i32
}

// -----

// CHECK-LABEL: @buffer_view_compute_range
func @buffer_view_compute_range(%arg0 : !iree.ref<!hal.buffer>) -> (i32, i32) {
  %0:2 = "test_hal.shape"() : () -> (i32, i32)
  %1:2 = "test_hal.indices"() : () -> (i32, i32)
  %2:2 = "test_hal.lengths"() : () -> (i32, i32)
  // CHECK: %off, %len = hal.buffer_view.compute_range %arg0, shape=[%0#0, %0#1], indices=[%1#0, %1#1], lengths=[%2#0, %2#1], element_size=4
  %off, %len = hal.buffer_view.compute_range %arg0, shape=[%0#0, %0#1], indices=[%1#0, %1#1], lengths=[%2#0, %2#1], element_size=4
  return %off, %len : i32, i32
}

// -----

// CHECK-LABEL: @buffer_view_slice
func @buffer_view_slice(%arg0 : !iree.ref<!hal.buffer>) -> !iree.ref<!hal.buffer> {
  %0:2 = "test_hal.shape"() : () -> (i32, i32)
  %1:2 = "test_hal.indices"() : () -> (i32, i32)
  %2:2 = "test_hal.lengths"() : () -> (i32, i32)
  // CHECK: %slice = hal.buffer_view.slice %arg0, shape=[%0#0, %0#1], indices=[%1#0, %1#1], lengths=[%2#0, %2#1], element_size=4
  %slice = hal.buffer_view.slice %arg0, shape=[%0#0, %0#1], indices=[%1#0, %1#1], lengths=[%2#0, %2#1], element_size=4
  return %slice : !iree.ref<!hal.buffer>
}

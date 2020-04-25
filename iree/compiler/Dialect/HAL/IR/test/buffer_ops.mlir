// Tests printing and parsing of hal.buffer ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @buffer_allocator
func @buffer_allocator() -> !hal.allocator {
  %0 = "test_hal.buffer"() : () -> !hal.buffer
  // CHECK: %allocator = hal.buffer.allocator %0 : !hal.allocator
  %allocator = hal.buffer.allocator %0 : !hal.allocator
  return %allocator : !hal.allocator
}

// -----

// CHECK-LABEL: @buffer_subspan
func @buffer_subspan() -> !hal.buffer {
  %0 = "test_hal.buffer"() : () -> !hal.buffer
  %1 = "test_hal.device_size"() : () -> index
  %2 = "test_hal.device_size"() : () -> index
  // CHECK: %buffer = hal.buffer.subspan %0, %1, %2 : !hal.buffer
  %buffer = hal.buffer.subspan %0, %1, %2 : !hal.buffer
  return %buffer : !hal.buffer
}

// -----

// CHECK-LABEL: @buffer_fill
func @buffer_fill(%arg0 : !hal.buffer) {
  %0 = "test_hal.device_size"() : () -> index
  %1 = "test_hal.device_size"() : () -> index
  %2 = "test_hal.pattern"() : () -> i32
  // CHECK: hal.buffer.fill %arg0, %0, %1, %2
  hal.buffer.fill %arg0, %0, %1, %2
  return
}

// -----

// CHECK-LABEL: @buffer_read_data
func @buffer_read_data(%arg0 : !hal.buffer) {
  %0 = "test_hal.device_size"() : () -> index
  %1 = "test_hal.mutable_data"() : () -> !iree.mutable_byte_buffer
  %2 = "test_hal.device_size"() : () -> index
  %3 = "test_hal.device_size"() : () -> index
  // CHECK: hal.buffer.read_data %arg0, %0, %1, %2, %3 : !iree.mutable_byte_buffer
  hal.buffer.read_data %arg0, %0, %1, %2, %3 : !iree.mutable_byte_buffer
  return
}

// -----

// CHECK-LABEL: @buffer_write_data
func @buffer_write_data(%arg0 : !hal.buffer) {
  %0 = "test_hal.mutable_data"() : () -> !iree.mutable_byte_buffer
  %1 = "test_hal.device_size"() : () -> index
  %2 = "test_hal.device_size"() : () -> index
  %3 = "test_hal.device_size"() : () -> index
  // CHECK: hal.buffer.write_data %0, %1, %arg0, %2, %3 : !iree.mutable_byte_buffer
  hal.buffer.write_data %0, %1, %arg0, %2, %3 : !iree.mutable_byte_buffer
  return
}

// -----

// CHECK-LABEL: @buffer_copy_data
func @buffer_copy_data(%arg0 : !hal.buffer, %arg1 : !hal.buffer) {
  %0 = "test_hal.device_size"() : () -> index
  %1 = "test_hal.device_size"() : () -> index
  %2 = "test_hal.device_size"() : () -> index
  // CHECK: hal.buffer.copy_data %arg0, %0, %arg1, %1, %2
  hal.buffer.copy_data %arg0, %0, %arg1, %1, %2
  return
}

// -----

// CHECK-LABEL: @buffer_load
func @buffer_load(%arg0 : !hal.buffer) -> i32 {
  %0 = "test_hal.device_size"() : () -> index
  // CHECK: %[[VAL:.+]] = hal.buffer.load %arg0[%0] : i32
  %1 = hal.buffer.load %arg0[%0] : i32
  // CHECK-NEXT: return %[[VAL]]
  return %1 : i32
}

// -----

// CHECK-LABEL: @buffer_store
func @buffer_store(%arg0 : i32, %arg1 : !hal.buffer) {
  %0 = "test_hal.device_size"() : () -> index
  // CHECK: hal.buffer.store %arg0, %arg1[%0] : i32
  hal.buffer.store %arg0, %arg1[%0] : i32
  return
}

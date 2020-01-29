// Tests printing and parsing of hal.buffer ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @buffer_allocator
func @buffer_allocator() -> !iree.ref<!hal.allocator> {
  %0 = "test_hal.buffer"() : () -> !iree.ref<!hal.buffer>
  // CHECK: %allocator = hal.buffer.allocator %0 : !iree.ref<!hal.allocator>
  %allocator = hal.buffer.allocator %0 : !iree.ref<!hal.allocator>
  return %allocator : !iree.ref<!hal.allocator>
}

// -----

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

// Tests printing and parsing of hal.allocator ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @allocator_compute_size
func @allocator_compute_size() -> i32 {
  %0 = "test_hal.allocator"() : () -> !hal.allocator
  %1:2 = "test_hal.shape"() : () -> (i32, i32)
  // CHECK: [[SZ:%.+]] = hal.allocator.compute_size %0, shape = [%1#0, %1#1], element_type = 32
  %sz = hal.allocator.compute_size %0, shape = [%1#0, %1#1], element_type = 32
  // CHECK-NEXT: return [[SZ]]
  return %sz : i32
}

// -----

// CHECK-LABEL: @allocator_compute_offset
func @allocator_compute_offset() -> i32 {
  %0 = "test_hal.allocator"() : () -> !hal.allocator
  %1:2 = "test_hal.shape"() : () -> (i32, i32)
  %2:2 = "test_hal.indices"() : () -> (i32, i32)
  // CHECK: %off = hal.allocator.compute_offset %0, shape = [%1#0, %1#1], element_type = 32, indices = [%2#0, %2#1]
  %off = hal.allocator.compute_offset %0, shape = [%1#0, %1#1], element_type = 32, indices = [%2#0, %2#1]
  return %off : i32
}

// -----

// CHECK-LABEL: @allocator_compute_range
func @allocator_compute_range() -> (i32, i32) {
  %0 = "test_hal.allocator"() : () -> !hal.allocator
  %1:2 = "test_hal.shape"() : () -> (i32, i32)
  %2:2 = "test_hal.indices"() : () -> (i32, i32)
  %3:2 = "test_hal.lengths"() : () -> (i32, i32)
  // CHECK: %off, %len = hal.allocator.compute_range %0, shape = [%1#0, %1#1], element_type = 32, indices = [%2#0, %2#1], lengths = [%3#0, %3#1]
  %off, %len = hal.allocator.compute_range %0, shape = [%1#0, %1#1], element_type = 32, indices = [%2#0, %2#1], lengths=[%3#0, %3#1]
  return %off, %len : i32, i32
}

// -----

// CHECK-LABEL: @allocator_allocate
func @allocator_allocate() -> !hal.buffer {
  // CHECK-DAG: [[C123:%.+]] = constant 123
  %0 = constant 123 : i32
  // CHECK-DAG: [[AL:%.+]] = "test_hal.allocator"
  %1 = "test_hal.allocator"() : () -> !hal.allocator
  // CHECK: [[CB:%.+]] = hal.allocator.allocate [[AL]], "HostVisible|HostCoherent", "Transfer", [[C123]] : !hal.buffer
  %buffer = hal.allocator.allocate %1, "HostVisible|HostCoherent", "Transfer", %0 : !hal.buffer
  // CHECK-NEXT: return [[CB]]
  return %buffer : !hal.buffer
}

// -----

// CHECK-LABEL: @allocator_allocate_const
func @allocator_allocate_const() -> !hal.buffer {
  // CHECK-DAG: [[AL:%.+]] = "test_hal.allocator"
  %allocator = "test_hal.allocator"() : () -> !hal.allocator
  // CHECK: [[CB:%.+]] = hal.allocator.allocate.const [[AL]], "HostVisible|HostCoherent", "Transfer" : !hal.buffer = dense<123> : tensor<4x4xi32>
  %buffer = hal.allocator.allocate.const %allocator, "HostVisible|HostCoherent", "Transfer" : !hal.buffer = dense<123> : tensor<4x4xi32>
  // CHECK-NEXT: return [[CB]]
  return %buffer : !hal.buffer
}

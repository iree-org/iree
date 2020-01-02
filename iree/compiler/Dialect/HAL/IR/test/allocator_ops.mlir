// Tests printing and parsing of hal.allocator ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @allocator_compute_size
func @allocator_compute_size() -> i32 {
  %0 = "test_hal.allocator"() : () -> !iree.ref<!hal.allocator>
  %1:2 = "test_hal.shape"() : () -> (i32, i32)
  // CHECK: [[SZ:%.+]] = hal.allocator.compute_size %0, "HostVisible|HostCoherent", "Transfer", shape=[%1#0, %1#1], element_size=4
  %sz = hal.allocator.compute_size %0, "HostVisible|HostCoherent", "Transfer", shape=[%1#0, %1#1], element_size=4
  // CHECK-NEXT: return [[SZ]]
  return %sz : i32
}

// -----

// CHECK-LABEL: @allocator_allocate
func @allocator_allocate() -> !iree.ref<!hal.buffer> {
  // CHECK-DAG: [[C123:%.+]] = constant 123
  %0 = constant 123 : i32
  // CHECK-DAG: [[AL:%.+]] = "test_hal.allocator"
  %1 = "test_hal.allocator"() : () -> !iree.ref<!hal.allocator>
  // CHECK: [[CB:%.+]] = hal.allocator.allocate [[AL]], "HostVisible|HostCoherent", "Transfer", [[C123]] : !iree.ref<!hal.buffer>
  %buffer = hal.allocator.allocate %1, "HostVisible|HostCoherent", "Transfer", %0 : !iree.ref<!hal.buffer>
  // CHECK-NEXT: return [[CB]]
  return %buffer : !iree.ref<!hal.buffer>
}

// -----

// CHECK-LABEL: @allocator_allocate_const
func @allocator_allocate_const() -> !iree.ref<!hal.buffer> {
  // CHECK-DAG: [[AL:%.+]] = "test_hal.allocator"
  %allocator = "test_hal.allocator"() : () -> !iree.ref<!hal.allocator>
  // CHECK: [[CB:%.+]] = hal.allocator.allocate.const [[AL]], "HostVisible|HostCoherent", "Transfer" : !iree.ref<!hal.buffer> = dense<123> : tensor<4x4xi32>
  %buffer = hal.allocator.allocate.const %allocator, "HostVisible|HostCoherent", "Transfer" : !iree.ref<!hal.buffer> = dense<123> : tensor<4x4xi32>
  // CHECK-NEXT: return [[CB]]
  return %buffer : !iree.ref<!hal.buffer>
}

// -----

// CHECK-LABEL: @allocator_allocate_shaped
func @allocator_allocate_shaped() -> !iree.ref<!hal.buffer> {
  // CHECK-DAG: [[AL:%.+]] = "test_hal.allocator"
  %0 = "test_hal.allocator"() : () -> !iree.ref<!hal.allocator>
  // CHECK-DAG: {{.+}} = "test_hal.shape"
  %1:2 = "test_hal.shape"() : () -> (i32, i32)
  // CHECK: [[CB:%.+]] = hal.allocator.allocate.shaped [[AL]], "HostVisible|HostCoherent", "Transfer", shape=[{{.+}}#0, {{.+}}#1], element_size=4 : !iree.ref<!hal.buffer>
  %buffer = hal.allocator.allocate.shaped %0, "HostVisible|HostCoherent", "Transfer", shape=[%1#0, %1#1], element_size=4 : !iree.ref<!hal.buffer>
  // CHECK-NEXT: return [[CB]]
  return %buffer : !iree.ref<!hal.buffer>
}

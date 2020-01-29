// Tests folding and canonicalization of HAL buffer ops.

// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @skip_buffer_allocator
func @skip_buffer_allocator() -> !iree.ref<!hal.allocator> {
  // CHECK-DAG: [[AL:%.+]] = "test_hal.allocator"
  %0 = "test_hal.allocator"() : () -> !iree.ref<!hal.allocator>
  %sz = constant 4 : i32
  %buffer = hal.allocator.allocate %0, "HostVisible|HostCoherent", "Transfer", %sz : !iree.ref<!hal.buffer>
  %1 = hal.buffer.allocator %buffer : !iree.ref<!hal.allocator>
  // CHECK: return [[AL]]
  return %1 : !iree.ref<!hal.allocator>
}

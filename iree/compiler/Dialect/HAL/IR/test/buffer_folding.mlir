// Tests folding and canonicalization of HAL buffer ops.

// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @skip_buffer_allocator
func @skip_buffer_allocator() -> !hal.allocator {
  // CHECK-DAG: [[AL:%.+]] = "test_hal.allocator"
  %0 = "test_hal.allocator"() : () -> !hal.allocator
  %sz = constant 4 : i32
  %buffer = hal.allocator.allocate %0, "HostVisible|HostCoherent", "Transfer", %sz : !hal.buffer
  %1 = hal.buffer.allocator %buffer : !hal.allocator
  // CHECK: return [[AL]]
  return %1 : !hal.allocator
}

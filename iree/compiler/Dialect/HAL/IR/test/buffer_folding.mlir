// Tests folding and canonicalization of HAL buffer ops.

// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @skip_buffer_allocator
func @skip_buffer_allocator() -> !hal.allocator {
  // CHECK-DAG: %[[AL:.+]] = "test_hal.allocator"
  %0 = "test_hal.allocator"() : () -> !hal.allocator
  %sz = constant 4 : index
  %buffer = hal.allocator.allocate %0, "HostVisible|HostCoherent", "Transfer", %sz : !hal.buffer
  %1 = hal.buffer.allocator %buffer : !hal.allocator
  // CHECK: return %[[AL]]
  return %1 : !hal.allocator
}

// -----

// CHECK-LABEL: @skip_subspan_buffer_allocator
func @skip_subspan_buffer_allocator() -> !hal.allocator {
  %c0 = constant 0 : index
  %c184 = constant 184 : index
  %c384 = constant 384 : index
  // CHECK-DAG: %[[AL:.+]] = "test_hal.allocator"
  %allocator = "test_hal.allocator"() : () -> !hal.allocator
  %source_buffer = hal.allocator.allocate %allocator, "HostVisible|HostCoherent", "Transfer", %c384 : !hal.buffer
  %span_buffer = hal.buffer.subspan %source_buffer, %c0, %c184 : !hal.buffer
  %1 = hal.buffer.allocator %span_buffer : !hal.allocator
  // CHECK: return %[[AL]]
  return %1 : !hal.allocator
}

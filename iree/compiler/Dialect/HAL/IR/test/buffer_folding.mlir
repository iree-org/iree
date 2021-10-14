// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @skip_buffer_allocator
//  CHECK-SAME: (%[[ALLOCATOR:.+]]: !hal.allocator)
func @skip_buffer_allocator(%allocator: !hal.allocator) -> !hal.allocator {
  %sz = arith.constant 4 : index
  %buffer = hal.allocator.allocate<%allocator : !hal.allocator>
                type("HostVisible|HostCoherent")
                usage(Transfer) : !hal.buffer{%sz}
  %1 = hal.buffer.allocator<%buffer : !hal.buffer> : !hal.allocator
  // CHECK: return %[[ALLOCATOR]]
  return %1 : !hal.allocator
}

// -----

// CHECK-LABEL: @skip_subspan_buffer_allocator
//  CHECK-SAME: (%[[ALLOCATOR:.+]]: !hal.allocator)
func @skip_subspan_buffer_allocator(%allocator: !hal.allocator) -> !hal.allocator {
  %c0 = arith.constant 0 : index
  %c184 = arith.constant 184 : index
  %c384 = arith.constant 384 : index
  %source_buffer = hal.allocator.allocate<%allocator : !hal.allocator>
                     type("HostVisible|HostCoherent")
                     usage(Transfer) : !hal.buffer{%c384}
  %span_buffer = hal.buffer.subspan<%source_buffer : !hal.buffer>[%c0, %c184] : !hal.buffer
  %1 = hal.buffer.allocator<%span_buffer : !hal.buffer> : !hal.allocator
  // CHECK: return %[[ALLOCATOR]]
  return %1 : !hal.allocator
}

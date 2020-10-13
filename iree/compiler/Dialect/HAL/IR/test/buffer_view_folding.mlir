// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @expand_buffer_view_const
func @expand_buffer_view_const() -> !hal.buffer_view {
  %0 = "test_hal.allocator"() : () -> !hal.allocator
  //      CHECK: [[CONST:%.+]] = iree.byte_buffer.constant : !iree.byte_buffer = dense<[4, 1, 2]> : tensor<3xi32>
  // CHECK-NEXT: [[BUFFER:%.+]] = hal.allocator.map {{.+}}, "HostVisible|HostCoherent", "Transfer", [[CONST]][%c0, %c-1] : !iree.byte_buffer -> !hal.buffer
  // CHECK-NEXT: [[VIEW:%.+]] = hal.buffer_view.create [[BUFFER]], shape = [%c3], element_type = 16777248 : !hal.buffer_view
  %view = hal.buffer_view.const %0, "HostVisible|HostCoherent", "Transfer" : !hal.buffer_view = dense<[4, 1, 2]> : tensor<3xi32>
  // CHECK-NEXT: return [[VIEW]]
  return %view : !hal.buffer_view
}

// -----

// CHECK-LABEL: @skip_buffer_view_buffer
func @skip_buffer_view_buffer() -> !hal.buffer {
  // CHECK: %[[BUFFER:.+]] = "test_hal.buffer"
  %0 = "test_hal.buffer"() : () -> !hal.buffer
  %1:2 = "test_hal.shape"() : () -> (index, index)
  %2 = hal.buffer_view.create %0, shape = [%1#0, %1#1], element_type = 32 : !hal.buffer_view
  %3 = hal.buffer_view.buffer %2 : !hal.buffer
  // CHECK: return %[[BUFFER]]
  return %3 : !hal.buffer
}

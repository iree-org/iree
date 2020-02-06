// Tests folding and canonicalization of HAL buffer view ops.

// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @expand_buffer_view_const
func @expand_buffer_view_const() -> !iree.ref<!hal.buffer_view> {
  %0 = "test_hal.allocator"() : () -> !iree.ref<!hal.allocator>
  // CHECK: [[BUFFER:%.+]] = hal.allocator.allocate.const %0, "HostVisible|HostCoherent", "Transfer" : !iree.ref<!hal.buffer> = dense<[4, 1, 2]> : tensor<3xi32>
  // CHECK-NEXT: [[VIEW:%.+]] = hal.buffer_view.create [[BUFFER]], shape=[%c3_i32], element_type=16777248 : !iree.ref<!hal.buffer_view>
  %view = hal.buffer_view.const %0, "HostVisible|HostCoherent", "Transfer" : !iree.ref<!hal.buffer_view> = dense<[4, 1, 2]> : tensor<3xi32>
  // CHECK-NEXT: return [[VIEW]]
  return %view : !iree.ref<!hal.buffer_view>
}

// -----

// CHECK-LABEL: @skip_buffer_view_buffer
func @skip_buffer_view_buffer() -> !iree.ref<!hal.buffer> {
  // CHECK: [[BUFFER:%.+]] = "test_hal.buffer"
  %0 = "test_hal.buffer"() : () -> !iree.ref<!hal.buffer>
  %1:2 = "test_hal.shape"() : () -> (i32, i32)
  %2 = hal.buffer_view.create %0, shape=[%1#0, %1#1], element_type=32 : !iree.ref<!hal.buffer_view>
  %3 = hal.buffer_view.buffer %2 : !iree.ref<!hal.buffer>
  // CHECK: return [[BUFFER]]
  return %3 : !iree.ref<!hal.buffer>
}

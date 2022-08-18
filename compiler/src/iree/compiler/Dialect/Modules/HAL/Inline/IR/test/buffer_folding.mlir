// RUN: iree-opt --split-input-file --canonicalize -cse %s | iree-opt --allow-unregistered-dialect --split-input-file | FileCheck %s

// CHECK-LABEL: func @fold_buffer_length
// CHECK-SAME: (%[[LENGTH:.+]]: index)
func.func @fold_buffer_length(%length: index) -> index {
  %c64 = arith.constant 64 : index
  %buffer, %storage = hal_inline.buffer.allocate alignment(%c64) : !hal.buffer{%length} in !util.buffer
  // CHECK-NOT: hal_inline.buffer.length
  %queried_length = hal_inline.buffer.length<%buffer : !hal.buffer> : index
  // CHECK: return %[[LENGTH]]
  return %queried_length : index
}

// -----

// CHECK-LABEL: func @fold_buffer_storage
func.func @fold_buffer_storage(%length: index) -> !util.buffer {
  %c64 = arith.constant 64 : index
  // CHECK: %[[BUFFER:.+]], %[[STORAGE:.+]] = hal_inline.buffer.allocate
  %buffer, %storage = hal_inline.buffer.allocate alignment(%c64) : !hal.buffer{%length} in !util.buffer
  // CHECK-NOT: hal_inline.buffer.storage
  %queried_storage = hal_inline.buffer.storage<%buffer : !hal.buffer> : !util.buffer
  // CHECK: return %[[STORAGE]]
  return %queried_storage : !util.buffer
}

// -----

// CHECK-LABEL: func @skip_buffer_view_buffer
// CHECK-SAME: (%[[BUFFER:.+]]: !hal.buffer)
func.func @skip_buffer_view_buffer(%buffer: !hal.buffer) -> !hal.buffer {
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %c32 = arith.constant 32 : i32
  %view = hal_inline.buffer_view.create buffer(%buffer : !hal.buffer)
                                        shape([%c10, %c11])
                                        type(%c32)
                                        encoding(%c1) : !hal.buffer_view
  %view_buffer = hal_inline.buffer_view.buffer<%view : !hal.buffer_view> : !hal.buffer
  // CHECK: return %[[BUFFER]]
  return %view_buffer : !hal.buffer
}

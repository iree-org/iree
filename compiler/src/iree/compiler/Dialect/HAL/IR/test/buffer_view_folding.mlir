// RUN: iree-opt --split-input-file --canonicalize -cse %s | iree-opt --allow-unregistered-dialect --split-input-file | FileCheck %s

// CHECK-LABEL: func @skip_buffer_view_buffer
// CHECK-SAME: %[[BUFFER:.+]]: !hal.buffer
func.func @skip_buffer_view_buffer(%buffer : !hal.buffer) -> !hal.buffer {
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %c32 = arith.constant 32 : i32
  %view = hal.buffer_view.create buffer(%buffer : !hal.buffer)
                                 shape([%c10, %c11])
                                 type(%c32)
                                 encoding(%c1) : !hal.buffer_view
  %view_buffer = hal.buffer_view.buffer<%view : !hal.buffer_view> : !hal.buffer
  // CHECK: return %[[BUFFER]]
  return %view_buffer : !hal.buffer
}

// -----

// CHECK-LABEL: func @expand_buffer_view_dims
// CHECK-SAME: %[[VIEW:.+]]: !hal.buffer_view
func.func @expand_buffer_view_dims(%view : !hal.buffer_view) -> (index, index, index) {
  // CHECK-DAG: %[[D0:.+]] = hal.buffer_view.dim<%[[VIEW]] : !hal.buffer_view>[0] : index
  // CHECK-DAG: %[[D1:.+]] = hal.buffer_view.dim<%[[VIEW]] : !hal.buffer_view>[1] : index
  // CHECK-DAG: %[[D2:.+]] = hal.buffer_view.dim<%[[VIEW]] : !hal.buffer_view>[2] : index
  %0, %1, %2 = hal.buffer_view.dims<%view : !hal.buffer_view> : index, index, index
  // CHECK-NEXT: return %[[D0]], %[[D1]], %[[D2]]
  return %0, %1, %2 : index, index, index
}

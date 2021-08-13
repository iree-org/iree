// RUN: iree-opt -allow-unregistered-dialect -split-input-file -canonicalize -cse %s | iree-opt -allow-unregistered-dialect -split-input-file | IreeFileCheck %s

// CHECK-LABEL: func @skip_buffer_view_buffer
// CHECK-SAME: %[[BUFFER:.+]]: !hal.buffer
func @skip_buffer_view_buffer(%buffer : !hal.buffer) -> !hal.buffer {
  %c10 = constant 10 : index
  %c11 = constant 11 : index
  %c32 = constant 32 : i32
  %view = hal.buffer_view.create %buffer, element_type = %c32, shape = [%c10, %c11] : !hal.buffer -> !hal.buffer_view
  %view_buffer = hal.buffer_view.buffer %view : !hal.buffer
  // CHECK: return %[[BUFFER]]
  return %view_buffer : !hal.buffer
}

// -----

// CHECK-LABEL: func @expand_buffer_view_dims
// CHECK-SAME: %[[VIEW:.+]]: !hal.buffer_view
func @expand_buffer_view_dims(%arg0 : !hal.buffer_view) -> (index, index, index) {
  // CHECK-DAG: %[[D0:.+]] = hal.buffer_view.dim %[[VIEW]], 0 : index
  // CHECK-DAG: %[[D1:.+]] = hal.buffer_view.dim %[[VIEW]], 1 : index
  // CHECK-DAG: %[[D2:.+]] = hal.buffer_view.dim %[[VIEW]], 2 : index
  %0, %1, %2 = hal.buffer_view.dims %arg0 : index, index, index
  // CHECK-NEXT: return %[[D0]], %[[D1]], %[[D2]]
  return %0, %1, %2 : index, index, index
}

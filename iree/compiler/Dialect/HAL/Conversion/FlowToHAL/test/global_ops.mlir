// RUN: iree-opt -split-input-file -iree-convert-to-hal -verify-diagnostics %s | IreeFileCheck %s

// CHECK-LABEL: util.global public mutable @var_i32 : !hal.buffer
util.global public mutable @var_i32 : tensor<i32>
func @fn() {
  // CHECK: %[[V:.+]] = util.global.load @var_i32 : !hal.buffer
  %0 = util.global.load @var_i32 : tensor<i32>
  // CHECK-NEXT: util.global.store %[[V]], @var_i32 : !hal.buffer
  util.global.store %0, @var_i32 : tensor<i32>
  return
}

// -----

// CHECK-LABEL: util.global public mutable @var_i1 : !hal.buffer
util.global public mutable @var_i1 : tensor<i1>
func @fn() {
  // CHECK: %[[V:.+]] = util.global.load @var_i1 : !hal.buffer
  %0 = util.global.load @var_i1 : tensor<i1>
  // CHECK-NEXT: util.global.store %[[V]], @var_i1 : !hal.buffer
  util.global.store %0, @var_i1 : tensor<i1>
  return
}

// -----

// CHECK-LABEL: util.global public mutable @var_indirect : !hal.buffer
util.global public mutable @var_indirect : tensor<i32>
func @fn() {
  // CHECK: %[[ADDR:.+]] = util.global.address @var_indirect
  %0 = util.global.address @var_indirect : !util.ptr<tensor<i32>>
  // CHECK-NEXT: %[[VALUE:.+]] = util.global.load.indirect %[[ADDR]]
  %1 = util.global.load.indirect %0 : !util.ptr<tensor<i32>> -> tensor<i32>
  // CHECK-NEXT: util.global.store.indirect %[[VALUE]], %[[ADDR]]
  util.global.store.indirect %1, %0 : tensor<i32> -> !util.ptr<tensor<i32>>
  return
}

// -----

// Checks that an initializer function is generated, used and operates on
// a hal.buffer (versus tensor).

// CHECK: util.global public mutable @var_with_tensor_initializer : !hal.buffer
util.global public mutable @var_with_tensor_initializer = dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT: util.initializer {
// CHECK: util.global.store %{{.+}}, @var_with_tensor_initializer : !hal.buffer
func @fn() {
  %0 = util.global.load @var_with_tensor_initializer : tensor<f32>
  util.global.store %0, @var_with_tensor_initializer : tensor<f32>
  return
}

// -----

// Checks that the implicit cast allowing a buffer_view to store into a variable
// that maps to a buffer is permitted.

// CHECK-LABEL: util.global public mutable @var_with_buffer_view_store
// CHECK: %[[BUFFER:.*]] = hal.buffer_view.buffer<%arg0 : !hal.buffer_view> : !hal.buffer
// CHECK: util.global.store %[[BUFFER]], @var_with_buffer_view_store : !hal.buffer
util.global public mutable @var_with_buffer_view_store = dense<0.000000e+00> : tensor<f32>
func @fn(%arg0: !hal.buffer_view) {
  %0 = hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<f32>
  util.global.store %0, @var_with_buffer_view_store : tensor<f32>
  return
}

// -----
// Checks that stores are permitted for variables that do not dominate the
// function containing a store.
// CHECK-LABEL: func @store_var_out_of_order
// CHECK: %[[BUFFER:.*]] = hal.buffer_view.buffer<%arg0 : !hal.buffer_view> : !hal.buffer
// CHECK: util.global.store %[[BUFFER]], @var_out_of_order : !hal.buffer
func @store_var_out_of_order(%arg0: !hal.buffer_view) {
  %0 = hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<f32>
  util.global.store %0, @var_out_of_order : tensor<f32>
  return
}
util.global public mutable @var_out_of_order = dense<0.000000e+00> : tensor<f32>

// -----
// Checks that the implicit cast allowing a buffer_view to indirect store into
// a variable that maps to a buffer is permitted.
// CHECK-LABEL: util.global public mutable @var_indirect_with_buffer_view_store
// CHECK: %[[PTR:.*]] = util.global.address @var_indirect_with_buffer_view_store : !util.ptr<!hal.buffer>
// CHECK: %[[BUFFER:.*]] = hal.buffer_view.buffer<%arg0 : !hal.buffer_view> : !hal.buffer
// CHECK: util.global.store.indirect %[[BUFFER]], %[[PTR]] : !hal.buffer -> !util.ptr<!hal.buffer>
util.global public mutable @var_indirect_with_buffer_view_store : tensor<i32>
func @fn(%arg0: !hal.buffer_view) {
  %0 = util.global.address @var_indirect_with_buffer_view_store : !util.ptr<tensor<i32>>
  %1 = hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<i32>
  util.global.store.indirect %1, %0 : tensor<i32> -> !util.ptr<tensor<i32>>
  return
}

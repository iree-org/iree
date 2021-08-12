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
// CHECK: util.global public mutable @var_with_tensor_default
// CHECK-SAME: initializer(@__var_with_tensor_default_initializer)
// CHECK-SAME: : !hal.buffer
// CHECK-LABEL: func @__var_with_tensor_default_initializer() -> !hal.buffer
util.global public mutable @var_with_tensor_default = dense<0.000000e+00> : tensor<f32>
func @fn() {
  %0 = util.global.load @var_with_tensor_default : tensor<f32>
  util.global.store %0, @var_with_tensor_default : tensor<f32>
  return
}

// -----

// TODO(b/145839814): It should not be possible to produce a name collision
// expected-error @+3 {{redefinition of symbol named '__var_with_initializer_initializer'}}
// expected-note @+1 {{see existing symbol definition here}}
func private @__var_with_initializer_initializer() -> ()
util.global public mutable @var_with_initializer = dense<0.000000e+00> : tensor<f32>
func @fn() {
  %0 = util.global.load @var_with_initializer : tensor<f32>
  util.global.store %0, @var_with_initializer : tensor<f32>
  return
}

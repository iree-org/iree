// RUN: iree-opt -split-input-file -iree-convert-flow-to-hal -verify-diagnostics %s | IreeFileCheck %s

// CHECK-LABEL: hal.variable @var_i32 mutable : !iree.ref<!hal.buffer>
flow.variable @var_i32 mutable : tensor<i32>
func @fn() {
  // CHECK: [[V:%.+]] = hal.variable.load @var_i32 : !iree.ref<!hal.buffer>
  %0 = flow.variable.load @var_i32 : tensor<i32>
  // CHECK-NEXT: hal.variable.store [[V]], @var_i32 : !iree.ref<!hal.buffer>
  flow.variable.store %0, @var_i32 : tensor<i32>
  return
}

// -----

// CHECK-LABEL: hal.variable @var_i1 mutable : !iree.ref<!hal.buffer>
flow.variable @var_i1 mutable : tensor<i1>
func @fn() {
  // CHECK: [[V:%.+]] = hal.variable.load @var_i1 : !iree.ref<!hal.buffer>
  %0 = flow.variable.load @var_i1 : tensor<i1>
  // CHECK-NEXT: hal.variable.store [[V]], @var_i1 : !iree.ref<!hal.buffer>
  flow.variable.store %0, @var_i1 : tensor<i1>
  return
}

// -----
// Checks that an initializer function is generated, used and operates on
// a hal.buffer (versus tensor).
// CHECK-LABEL: func @__var_with_tensor_initializer_initializer() -> !iree.ref<!hal.buffer>
// CHECK: hal.variable @var_with_tensor_initializer
// CHECK-SAME: init(@__var_with_tensor_initializer_initializer)
// CHECK-SAME: : !iree.ref<!hal.buffer>
flow.variable @var_with_tensor_initializer mutable dense<0.000000e+00> : tensor<f32>
func @fn() {
  %0 = flow.variable.load @var_with_tensor_initializer : tensor<f32>
  flow.variable.store %0, @var_with_tensor_initializer : tensor<f32>
  return
}

// -----
// TODO(b/145839814): It should not be possible to produce a name collision
// expected-error @+3 {{redefinition of symbol named '__var_with_initializer_initializer'}}
// expected-note @+1 {{see existing symbol definition here}}
func @__var_with_initializer_initializer() -> ()
flow.variable @var_with_initializer mutable dense<0.000000e+00> : tensor<f32>
func @fn() {
  %0 = flow.variable.load @var_with_initializer : tensor<f32>
  flow.variable.store %0, @var_with_initializer : tensor<f32>
  return
}

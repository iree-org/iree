// Tests printing and parsing of variable ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK: flow.variable @v_immutable : tensor<i32>
flow.variable @v_immutable : tensor<i32>
// CHECK: flow.variable @v_mutable mutable : tensor<i32>
flow.variable @v_mutable mutable : tensor<i32>

// -----

// CHECK: flow.variable @v_initialized_const dense<4> : tensor<4xi32>
flow.variable @v_initialized_const dense<4> : tensor<4xi32>

// -----

// CHECK: flow.variable @v_initialized init(@initializer) : tensor<4xi32>
flow.variable @v_initialized init(@initializer) : tensor<4xi32>
func @initializer() -> tensor<4xi32>

// -----

flow.variable @v_loaded : tensor<4xi32>
// CHECK-LABEL: @loaded
func @loaded() {
  // CHECK-NEXT: = flow.variable.load @v_loaded : tensor<4xi32>
  %0 = flow.variable.load @v_loaded : tensor<4xi32>
  return
}

// -----

flow.variable @v_stored mutable : tensor<4xi32>
// CHECK-LABEL: @stored
func @stored() {
  // CHECK-NEXT: [[VAL:%.+]] = constant
  %cst = constant dense<5> : tensor<4xi32>
  // CHECK-NEXT: flow.variable.store [[VAL]], @v_stored : tensor<4xi32>
  flow.variable.store %cst, @v_stored : tensor<4xi32>
  return
}

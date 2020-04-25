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
  // CHECK-NEXT: %[[VAL:.+]] = constant
  %cst = constant dense<5> : tensor<4xi32>
  // CHECK-NEXT: flow.variable.store %[[VAL]], @v_stored : tensor<4xi32>
  flow.variable.store %cst, @v_stored : tensor<4xi32>
  return
}

// -----

flow.variable @v_loaded : tensor<4xf32>
// CHECK-LABEL: @loaded_indirect
func @loaded_indirect() {
  // CHECK-NEXT: %[[ADDR:.+]] = flow.variable.address @v_loaded
  %0 = flow.variable.address @v_loaded : !iree.ptr<tensor<4xf32>>
  // CHECK-NEXT: = flow.variable.load.indirect %[[ADDR]]
  %1 = flow.variable.load.indirect %0 : !iree.ptr<tensor<4xf32>> -> tensor<4xf32>
  return
}

// -----

flow.variable @v_stored mutable : tensor<4xf32>
// CHECK-LABEL: @stored_indirect
func @stored_indirect() {
  // CHECK-NEXT: %[[VALUE:.+]] = "test_flow.tensor"
  %0 = "test_flow.tensor"() : () -> tensor<4xf32>
  // CHECK-NEXT: %[[ADDR:.+]] = flow.variable.address @v_stored
  %1 = flow.variable.address @v_stored : !iree.ptr<tensor<4xf32>>
  // CHECK-NEXT: flow.variable.store.indirect %[[VALUE]], %[[ADDR]]
  flow.variable.store.indirect %0, %1 : tensor<4xf32> -> !iree.ptr<tensor<4xf32>>
  return
}

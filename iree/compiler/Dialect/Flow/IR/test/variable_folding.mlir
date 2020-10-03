// Tests folding and canonicalization of variable ops.

// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK: flow.variable @v_initialized dense<4> : tensor<4xi32>
flow.variable @v_initialized init(@initializer) : tensor<4xi32>
func @initializer() -> tensor<4xi32> {
  %0 = constant dense<4> : tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

flow.variable @v_unused : tensor<4xi32>
// CHECK-LABEL: @unused_load
func @unused_load() {
  // CHECK-NEXT: return
  %0 = flow.variable.load @v_unused : tensor<4xi32>
  return
}

// -----

flow.variable @v_const dense<1.0> : tensor<8xf32>
// CHECK-LABEL: @fold_immutable_const
func @fold_immutable_const() -> tensor<8xf32> {
  // CHECK-NEXT: %[[CONST:.+]] = constant dense<1.{{.+}}> : tensor<8xf32>
  %0 = flow.variable.load @v_const : tensor<8xf32>
  // CHECK-NEXT: return %[[CONST]] : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

flow.variable @v_const dense<1.0> : tensor<8xf32> attributes {noinline}
// CHECK-LABEL: @no_fold_noinline_immutable_const
func @no_fold_noinline_immutable_const() -> tensor<8xf32> {
  // CHECK-NEXT: = flow.variable.load @v_const : tensor<8xf32>
  %0 = flow.variable.load @v_const : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

flow.variable @v_nop mutable : tensor<4xi32>
// CHECK-LABEL: @nop_load_store
func @nop_load_store() {
  // CHECK-NEXT: return
  %0 = flow.variable.load @v_nop : tensor<4xi32>
  flow.variable.store %0, @v_nop : tensor<4xi32>
  return
}

// -----

flow.variable @v : tensor<4xf32>
// CHECK-LABEL: @fold_load_indirect
func @fold_load_indirect() -> tensor<4xf32> {
  %0 = flow.variable.address @v : !iree.ptr<tensor<4xf32>>
  // CHECK-NEXT: = flow.variable.load @v
  %1 = flow.variable.load.indirect %0 : !iree.ptr<tensor<4xf32>> -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

flow.variable @v mutable : tensor<4xf32>
// CHECK-LABEL: @fold_store_indirect
func @fold_store_indirect(%arg0 : tensor<4xf32>) {
  %0 = flow.variable.address @v : !iree.ptr<tensor<4xf32>>
  // CHECK-NEXT: flow.variable.store %arg0, @v
  flow.variable.store.indirect %arg0, %0 : tensor<4xf32> -> !iree.ptr<tensor<4xf32>>
  return
}

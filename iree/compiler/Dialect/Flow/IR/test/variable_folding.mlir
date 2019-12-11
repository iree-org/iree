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

flow.variable @v_nop mutable : tensor<4xi32>
// CHECK-LABEL: @nop_load_store
func @nop_load_store() {
  // CHECK-NEXT: return
  %0 = flow.variable.load @v_nop : tensor<4xi32>
  flow.variable.store %0, @v_nop : tensor<4xi32>
  return
}


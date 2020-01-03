// Tests folding and canonicalization of variable ops.

// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK: hal.variable @v_initialized 4 : i32
hal.variable @v_initialized init(@initializer) : i32
func @initializer() -> i32 {
  %0 = constant 4 : i32
  return %0 : i32
}

// -----

hal.variable @v_unused : !iree.ref<!hal.buffer>
// CHECK-LABEL: @unused_load
func @unused_load() {
  // CHECK-NEXT: return
  %0 = hal.variable.load @v_unused : !iree.ref<!hal.buffer>
  return
}

// -----

hal.variable @v_nop mutable : !iree.ref<!hal.buffer>
// CHECK-LABEL: @nop_load_store
func @nop_load_store() {
  // CHECK-NEXT: return
  %0 = hal.variable.load @v_nop : !iree.ref<!hal.buffer>
  hal.variable.store %0, @v_nop : !iree.ref<!hal.buffer>
  return
}


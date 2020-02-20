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

// -----

hal.variable @v : !iree.ref<!hal.buffer>
// CHECK-LABEL: @fold_load_indirect
func @fold_load_indirect() -> !iree.ref<!hal.buffer> {
  %0 = hal.variable.address @v : !iree.ptr<!iree.ref<!hal.buffer>>
  // CHECK-NEXT: = hal.variable.load @v
  %1 = hal.variable.load.indirect %0 : !iree.ptr<!iree.ref<!hal.buffer>> -> !iree.ref<!hal.buffer>
  return %1 : !iree.ref<!hal.buffer>
}

// -----

hal.variable @v mutable : !iree.ref<!hal.buffer>
// CHECK-LABEL: @fold_store_indirect
func @fold_store_indirect(%arg0 : !iree.ref<!hal.buffer>) {
  %0 = hal.variable.address @v : !iree.ptr<!iree.ref<!hal.buffer>>
  // CHECK-NEXT: hal.variable.store %arg0, @v
  hal.variable.store.indirect %arg0, %0 : !iree.ref<!hal.buffer> -> !iree.ptr<!iree.ref<!hal.buffer>>
  return
}

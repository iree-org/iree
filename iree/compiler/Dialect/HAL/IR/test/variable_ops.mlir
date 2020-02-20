// Tests printing and parsing of variable ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK: hal.variable @v_immutable : tensor<i32>
hal.variable @v_immutable : tensor<i32>
// CHECK: hal.variable @v_mutable mutable : tensor<i32>
hal.variable @v_mutable mutable : tensor<i32>

// -----

// CHECK: hal.variable @v_initialized_const 4 : i32
hal.variable @v_initialized_const 4 : i32

// -----

// CHECK: hal.variable @v_initialized init(@initializer) : !iree.ref<!hal.buffer>
hal.variable @v_initialized init(@initializer) : !iree.ref<!hal.buffer>
func @initializer() -> !iree.ref<!hal.buffer>

// -----

hal.variable @v_loaded : !iree.ref<!hal.buffer>
// CHECK-LABEL: @loaded
func @loaded() {
  // CHECK-NEXT: = hal.variable.load @v_loaded : !iree.ref<!hal.buffer>
  %0 = hal.variable.load @v_loaded : !iree.ref<!hal.buffer>
  return
}

// -----

hal.variable @v_stored mutable : !iree.ref<!hal.buffer>
// CHECK-LABEL: @stored
func @stored() {
  // CHECK-NEXT: [[BUF:%.+]] = "test_hal.buffer"
  %0 = "test_hal.buffer"() : () -> !iree.ref<!hal.buffer>
  // CHECK-NEXT: hal.variable.store [[BUF]], @v_stored : !iree.ref<!hal.buffer>
  hal.variable.store %0, @v_stored : !iree.ref<!hal.buffer>
  return
}

// -----

hal.variable @v_loaded : !iree.ref<!hal.buffer>
// CHECK-LABEL: @loaded_indirect
func @loaded_indirect() {
  // CHECK-NEXT: [[ADDR:%.+]] = hal.variable.address @v_loaded
  %0 = hal.variable.address @v_loaded : !iree.ptr<!iree.ref<!hal.buffer>>
  // CHECK-NEXT: = hal.variable.load.indirect [[ADDR]]
  %1 = hal.variable.load.indirect %0 : !iree.ptr<!iree.ref<!hal.buffer>> -> !iree.ref<!hal.buffer>
  return
}

// -----

hal.variable @v_stored mutable : !iree.ref<!hal.buffer>
// CHECK-LABEL: @stored_indirect
func @stored_indirect() {
  // CHECK-NEXT: [[BUF:%.+]] = "test_hal.buffer"
  %0 = "test_hal.buffer"() : () -> !iree.ref<!hal.buffer>
  // CHECK-NEXT: [[ADDR:%.+]] = hal.variable.address @v_stored
  %1 = hal.variable.address @v_stored : !iree.ptr<!iree.ref<!hal.buffer>>
  // CHECK-NEXT: hal.variable.store.indirect [[BUF]], [[ADDR]]
  hal.variable.store.indirect %0, %1 : !iree.ref<!hal.buffer> -> !iree.ptr<!iree.ref<!hal.buffer>>
  return
}

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

// CHECK: hal.variable @v_initialized init(@initializer) : !ireex.ref<!hal.buffer>
hal.variable @v_initialized init(@initializer) : !ireex.ref<!hal.buffer>
func @initializer() -> !ireex.ref<!hal.buffer>

// -----

hal.variable @v_loaded : !ireex.ref<!hal.buffer>
// CHECK-LABEL: @loaded
func @loaded() {
  // CHECK-NEXT: = hal.variable.load @v_loaded : !ireex.ref<!hal.buffer>
  %0 = hal.variable.load @v_loaded : !ireex.ref<!hal.buffer>
  return
}

// -----

hal.variable @v_stored mutable : !ireex.ref<!hal.buffer>
// CHECK-LABEL: @stored
func @stored() {
  // CHECK-NEXT: [[BUF:%.+]] = "test_hal.buffer"
  %0 = "test_hal.buffer"() : () -> !ireex.ref<!hal.buffer>
  // CHECK-NEXT: hal.variable.store [[BUF]], @v_stored : !ireex.ref<!hal.buffer>
  hal.variable.store %0, @v_stored : !ireex.ref<!hal.buffer>
  return
}

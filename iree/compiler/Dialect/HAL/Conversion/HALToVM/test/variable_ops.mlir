// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK: vm.global.i32 @v_initialized_const 4 : i32
hal.variable @v_initialized_const 4 : i32

// -----

// CHECK: vm.global.ref @v_initialized init(@initializer) : !iree.ref<!hal.buffer>
hal.variable @v_initialized init(@initializer) : !iree.ref<!hal.buffer>
func @initializer() -> !iree.ref<!hal.buffer>

// -----

// CHECK: vm.global.ref @v_loaded : !iree.ref<!hal.buffer>
hal.variable @v_loaded : !iree.ref<!hal.buffer>
// CHECK-LABEL: func @loaded
func @loaded() {
  // CHECK: %v_loaded = vm.global.load.ref @v_loaded : !iree.ref<!hal.buffer>
  %0 = hal.variable.load @v_loaded : !iree.ref<!hal.buffer>
  return
}

// -----

// CHECK: vm.global.ref @v_stored mutable : !iree.ref<!hal.buffer>
hal.variable @v_stored mutable : !iree.ref<!hal.buffer>
// CHECK-LABEL: func @stored
func @stored() {
  %0 = "test_hal.buffer"() : () -> !iree.ref<!hal.buffer>
  // CHECK: vm.global.store.ref %0, @v_stored : !iree.ref<!hal.buffer>
  hal.variable.store %0, @v_stored : !iree.ref<!hal.buffer>
  return
}

// -----

hal.variable @v_loaded : !iree.ref<!hal.buffer>
// CHECK-LABEL: @loaded_indirect
func @loaded_indirect() -> !iree.ref<!hal.buffer> {
  // CHECK-NEXT: [[ADDR:%.+]] = vm.global.address @v_loaded
  %0 = hal.variable.address @v_loaded : !iree.ptr<!iree.ref<!hal.buffer>>
  // CHECK-NEXT: = vm.global.load.indirect.ref [[ADDR]]
  %1 = hal.variable.load.indirect %0 : !iree.ptr<!iree.ref<!hal.buffer>> -> !iree.ref<!hal.buffer>
  return %1 : !iree.ref<!hal.buffer>
}

// -----

hal.variable @v_stored mutable : !iree.ref<!hal.buffer>
// CHECK-LABEL: @stored_indirect
func @stored_indirect(%arg0 : !iree.ref<!hal.buffer>) {
  // CHECK-NEXT: [[ADDR:%.+]] = vm.global.address @v_stored
  %0 = hal.variable.address @v_stored : !iree.ptr<!iree.ref<!hal.buffer>>
  // CHECK-NEXT: vm.global.store.indirect.ref %arg0, [[ADDR]]
  hal.variable.store.indirect %arg0, %0 : !iree.ref<!hal.buffer> -> !iree.ptr<!iree.ref<!hal.buffer>>
  return
}

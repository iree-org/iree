// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK: vm.global.i32 @v_initialized_const 4 : i32
hal.variable @v_initialized_const = 4 : i32

// -----

// CHECK: vm.global.ref @v_initialized init(@initializer) : !vm.ref<!hal.buffer>
hal.variable @v_initialized init(@initializer) : !hal.buffer
func @initializer() -> !hal.buffer

// -----

// CHECK: vm.global.ref @v_loaded : !vm.ref<!hal.buffer>
hal.variable @v_loaded : !hal.buffer
// CHECK-LABEL: func @loaded
func @loaded() {
  // CHECK: %v_loaded = vm.global.load.ref @v_loaded : !vm.ref<!hal.buffer>
  %0 = hal.variable.load @v_loaded : !hal.buffer
  return
}

// -----

// CHECK: vm.global.ref @v_stored mutable : !vm.ref<!hal.buffer>
hal.variable @v_stored mutable : !hal.buffer
// CHECK-LABEL: func @stored
func @stored(%arg0 : !hal.buffer) {
  // CHECK: vm.global.store.ref %arg0, @v_stored : !vm.ref<!hal.buffer>
  hal.variable.store %arg0, @v_stored : !hal.buffer
  return
}

// -----

hal.variable @v_loaded : !hal.buffer
// CHECK-LABEL: @loaded_indirect
func @loaded_indirect() -> !hal.buffer {
  // CHECK-NEXT: %[[ADDR:.+]] = vm.global.address @v_loaded
  %0 = hal.variable.address @v_loaded : !iree.ptr<!hal.buffer>
  // CHECK-NEXT: = vm.global.load.indirect.ref %[[ADDR]]
  %1 = hal.variable.load.indirect %0 : !iree.ptr<!hal.buffer> -> !hal.buffer
  return %1 : !hal.buffer
}

// -----

hal.variable @v_stored mutable : !hal.buffer
// CHECK-LABEL: @stored_indirect
func @stored_indirect(%arg0 : !hal.buffer) {
  // CHECK-NEXT: %[[ADDR:.+]] = vm.global.address @v_stored
  %0 = hal.variable.address @v_stored : !iree.ptr<!hal.buffer>
  // CHECK-NEXT: vm.global.store.indirect.ref %arg0, %[[ADDR]]
  hal.variable.store.indirect %arg0, %0 : !hal.buffer -> !iree.ptr<!hal.buffer>
  return
}

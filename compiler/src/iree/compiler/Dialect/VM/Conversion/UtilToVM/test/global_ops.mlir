// RUN: iree-opt --split-input-file --iree-vm-conversion %s | FileCheck %s

// CHECK: vm.global.i32 public @v_initialized_const = 4 : i32
util.global @v_initialized_const = 4 : i32

// CHECK: vm.global.i32 private @v_private_const = 5 : i32
util.global private @v_private_const = 5 : i32

// CHECK: vm.global.i32 private @v_uninitialized : i32
util.global private @v_uninitialized = #util.uninitialized : i32

// CHECK: vm.global.ref private @v_uninitialized_ref : !vm.ref<!hal.buffer>
util.global private @v_uninitialized_ref = #util.uninitialized : !vm.ref<!hal.buffer>

// -----

// CHECK: vm.global.ref public @v_initialized : !vm.ref<!hal.buffer>
util.global public @v_initialized : !hal.buffer
// CHECK-NEXT: vm.initializer {
// CHECK-NEXT:   %[[REF:.+]] = vm.call @initializer() : () -> !vm.ref<!hal.buffer>
// CHECK-NEXT:   vm.global.store.ref %[[REF]], @v_initialized : !vm.ref<!hal.buffer>
// CHECK-NEXT:   vm.return
// CHECK-NEXT: }
util.initializer {
  %0 = func.call @initializer() : () -> !hal.buffer
  util.global.store %0, @v_initialized : !hal.buffer
  util.return
}
// CHECK-NEXT: vm.import private @initializer() -> !vm.ref<!hal.buffer>
func.func private @initializer() -> !hal.buffer

// -----

// CHECK: vm.global.ref public @v_loaded : !vm.ref<!hal.buffer>
util.global public @v_loaded : !hal.buffer
// CHECK-LABEL: vm.func private @loaded
func.func @loaded() {
  // CHECK: %v_loaded = vm.global.load.ref @v_loaded : !vm.ref<!hal.buffer>
  %0 = util.global.load @v_loaded : !hal.buffer
  return
}

// -----

// CHECK: vm.global.ref public mutable @v_stored : !vm.ref<!hal.buffer>
util.global public mutable @v_stored : !hal.buffer
// CHECK-LABEL: vm.func private @stored
func.func @stored(%arg0 : !hal.buffer) {
  // CHECK: vm.global.store.ref %arg0, @v_stored : !vm.ref<!hal.buffer>
  util.global.store %arg0, @v_stored : !hal.buffer
  return
}

// -----

util.global @v_loaded : !hal.buffer
// CHECK-LABEL: @loaded_indirect
func.func @loaded_indirect() -> !hal.buffer {
  // CHECK-NEXT: %[[ADDR:.+]] = vm.global.address @v_loaded
  %0 = util.global.address @v_loaded : !util.ptr<!hal.buffer>
  // CHECK-NEXT: = vm.global.load.indirect.ref %[[ADDR]]
  %1 = util.global.load.indirect %0 : !util.ptr<!hal.buffer> -> !hal.buffer
  return %1 : !hal.buffer
}

// -----

util.global mutable @v_stored : !hal.buffer
// CHECK-LABEL: @stored_indirect
func.func @stored_indirect(%arg0 : !hal.buffer) {
  // CHECK-NEXT: %[[ADDR:.+]] = vm.global.address @v_stored
  %0 = util.global.address @v_stored : !util.ptr<!hal.buffer>
  // CHECK-NEXT: vm.global.store.indirect.ref %arg0, %[[ADDR]]
  util.global.store.indirect %arg0, %0 : !hal.buffer -> !util.ptr<!hal.buffer>
  return
}

// Tests printing and parsing of global ops.

// RUN: iree-opt -split-input-file %s | FileCheck %s --dump-input=fail

// CHECK-LABEL: @global_load_i32
vm.module @my_module {
  vm.global.i32 @g0 : i32
  vm.func @global_load_i32() -> i32 {
    // CHECK: %g0 = vm.global.load.i32 @g0 : i32
    %g0 = vm.global.load.i32 @g0 : i32
    vm.return %g0 : i32
  }
}

// -----

// CHECK-LABEL: @global_store_i32
vm.module @my_module {
  vm.global.i32 @g0 mutable : i32
  vm.func @global_store_i32(%arg0 : i32) {
    // CHECK: vm.global.store.i32 @g0, %arg0 : i32
    vm.global.store.i32 @g0, %arg0 : i32
    vm.return
  }
}

// -----

// CHECK-LABEL: @global_load_ref
vm.module @my_module {
  vm.global.ref @g0 : !ireex.opaque_ref
  vm.func @global_load_ref() -> !ireex.opaque_ref {
    // CHECK: %g0 = vm.global.load.ref @g0 : !ireex.opaque_ref
    %g0 = vm.global.load.ref @g0 : !ireex.opaque_ref
    vm.return %g0 : !ireex.opaque_ref
  }
}

// -----

// CHECK-LABEL: @global_store_ref
vm.module @my_module {
  vm.global.ref @g0 mutable : !ireex.opaque_ref
  vm.func @global_store_ref(%arg0 : !ireex.opaque_ref) {
    // CHECK: vm.global.store.ref @g0, %arg0 : !ireex.opaque_ref
    vm.global.store.ref @g0, %arg0 : !ireex.opaque_ref
    vm.return
  }
}

// -----

// CHECK-LABEL: @global_reset_ref
vm.module @my_module {
  vm.global.ref @g0 mutable : !ireex.opaque_ref
  vm.func @global_reset_ref() {
    // CHECK: vm.global.reset.ref @g0
    vm.global.reset.ref @g0
    vm.return
  }
}

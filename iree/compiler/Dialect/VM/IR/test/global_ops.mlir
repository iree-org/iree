// Tests printing and parsing of global ops.

// RUN: iree-opt -split-input-file %s | IreeFileCheck %s

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
    // CHECK: vm.global.store.i32 %arg0, @g0 : i32
    vm.global.store.i32 %arg0, @g0 : i32
    vm.return
  }
}

// -----

// CHECK-LABEL: @global_load_indirect_i32
vm.module @my_module {
  vm.global.i32 @g0 : i32
  vm.func @global_load_indirect_i32() -> i32 {
    // CHECK: [[ADDR:%.+]] = vm.global.address @g0 : !iree.ptr<i32>
    %0 = vm.global.address @g0 : !iree.ptr<i32>
    // CHECK-NEXT: = vm.global.load.indirect.i32 [[ADDR]] : !iree.ptr<i32> -> i32
    %1 = vm.global.load.indirect.i32 %0 : !iree.ptr<i32> -> i32
    vm.return %1 : i32
  }
}

// -----

// CHECK-LABEL: @global_store_indirect_i32
vm.module @my_module {
  vm.global.i32 @g0 mutable : i32
  vm.func @global_store_indirect_i32(%arg0 : i32) {
    // CHECK: [[ADDR:%.+]] = vm.global.address @g0 : !iree.ptr<i32>
    %0 = vm.global.address @g0 : !iree.ptr<i32>
    // CHECK-NEXT: vm.global.store.indirect.i32 %arg0, [[ADDR]] : i32 -> !iree.ptr<i32>
    vm.global.store.indirect.i32 %arg0, %0 : i32 -> !iree.ptr<i32>
    vm.return
  }
}

// -----

// CHECK-LABEL: @global_load_ref
vm.module @my_module {
  vm.global.ref @g0 : !iree.opaque_ref
  vm.func @global_load_ref() -> !iree.opaque_ref {
    // CHECK: %g0 = vm.global.load.ref @g0 : !iree.opaque_ref
    %g0 = vm.global.load.ref @g0 : !iree.opaque_ref
    vm.return %g0 : !iree.opaque_ref
  }
}

// -----

// CHECK-LABEL: @global_store_ref
vm.module @my_module {
  vm.global.ref @g0 mutable : !iree.opaque_ref
  vm.func @global_store_ref(%arg0 : !iree.opaque_ref) {
    // CHECK: vm.global.store.ref %arg0, @g0 : !iree.opaque_ref
    vm.global.store.ref %arg0, @g0 : !iree.opaque_ref
    vm.return
  }
}

// -----

// CHECK-LABEL: @global_load_indirect_ref
vm.module @my_module {
  vm.global.ref @g0 : !iree.opaque_ref
  vm.func @global_load_indirect_ref() -> !iree.opaque_ref {
    // CHECK: [[ADDR:%.+]] = vm.global.address @g0 : !iree.ptr<!iree.opaque_ref>
    %0 = vm.global.address @g0 : !iree.ptr<!iree.opaque_ref>
    // CHECK-NEXT: = vm.global.load.indirect.ref [[ADDR]] : !iree.ptr<!iree.opaque_ref> -> !iree.opaque_ref
    %1 = vm.global.load.indirect.ref %0 : !iree.ptr<!iree.opaque_ref> -> !iree.opaque_ref
    vm.return %1 : !iree.opaque_ref
  }
}

// -----

// CHECK-LABEL: @global_store_indirect_ref
vm.module @my_module {
  vm.global.ref @g0 mutable : !iree.opaque_ref
  vm.func @global_store_indirect_ref(%arg0 : !iree.opaque_ref) {
    // CHECK: [[ADDR:%.+]] = vm.global.address @g0 : !iree.ptr<!iree.opaque_ref>
    %0 = vm.global.address @g0 : !iree.ptr<!iree.opaque_ref>
    // CHECK-NEXT: vm.global.store.indirect.ref %arg0, [[ADDR]] : !iree.opaque_ref -> !iree.ptr<!iree.opaque_ref>
    vm.global.store.indirect.ref %arg0, %0 : !iree.opaque_ref -> !iree.ptr<!iree.opaque_ref>
    vm.return
  }
}

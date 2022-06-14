// RUN: iree-opt --split-input-file %s | FileCheck %s

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
  vm.global.i32 mutable @g0 : i32
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
    // CHECK: %[[ADDR:.+]] = vm.global.address @g0 : !util.ptr<i32>
    %0 = vm.global.address @g0 : !util.ptr<i32>
    // CHECK-NEXT: = vm.global.load.indirect.i32 %[[ADDR]] : !util.ptr<i32> -> i32
    %1 = vm.global.load.indirect.i32 %0 : !util.ptr<i32> -> i32
    vm.return %1 : i32
  }
}

// -----

// CHECK-LABEL: @global_store_indirect_i32
vm.module @my_module {
  vm.global.i32 mutable @g0 : i32
  vm.func @global_store_indirect_i32(%arg0 : i32) {
    // CHECK: %[[ADDR:.+]] = vm.global.address @g0 : !util.ptr<i32>
    %0 = vm.global.address @g0 : !util.ptr<i32>
    // CHECK-NEXT: vm.global.store.indirect.i32 %arg0, %[[ADDR]] : i32 -> !util.ptr<i32>
    vm.global.store.indirect.i32 %arg0, %0 : i32 -> !util.ptr<i32>
    vm.return
  }
}

// -----

// CHECK-LABEL: @global_load_ref
vm.module @my_module {
  vm.global.ref @g0 : !vm.ref<?>
  vm.func @global_load_ref() -> !vm.ref<?> {
    // CHECK: %g0 = vm.global.load.ref @g0 : !vm.ref<?>
    %g0 = vm.global.load.ref @g0 : !vm.ref<?>
    vm.return %g0 : !vm.ref<?>
  }
}

// -----

// CHECK-LABEL: @global_store_ref
vm.module @my_module {
  vm.global.ref mutable @g0 : !vm.ref<?>
  vm.func @global_store_ref(%arg0 : !vm.ref<?>) {
    // CHECK: vm.global.store.ref %arg0, @g0 : !vm.ref<?>
    vm.global.store.ref %arg0, @g0 : !vm.ref<?>
    vm.return
  }
}

// -----

// CHECK-LABEL: @global_load_indirect_ref
vm.module @my_module {
  vm.global.ref @g0 : !vm.ref<?>
  vm.func @global_load_indirect_ref() -> !vm.ref<?> {
    // CHECK: %[[ADDR:.+]] = vm.global.address @g0 : !util.ptr<!vm.ref<?>>
    %0 = vm.global.address @g0 : !util.ptr<!vm.ref<?>>
    // CHECK-NEXT: = vm.global.load.indirect.ref %[[ADDR]] : !util.ptr<!vm.ref<?>> -> !vm.ref<?>
    %1 = vm.global.load.indirect.ref %0 : !util.ptr<!vm.ref<?>> -> !vm.ref<?>
    vm.return %1 : !vm.ref<?>
  }
}

// -----

// CHECK-LABEL: @global_store_indirect_ref
vm.module @my_module {
  vm.global.ref mutable @g0 : !vm.ref<?>
  vm.func @global_store_indirect_ref(%arg0 : !vm.ref<?>) {
    // CHECK: %[[ADDR:.+]] = vm.global.address @g0 : !util.ptr<!vm.ref<?>>
    %0 = vm.global.address @g0 : !util.ptr<!vm.ref<?>>
    // CHECK-NEXT: vm.global.store.indirect.ref %arg0, %[[ADDR]] : !vm.ref<?> -> !util.ptr<!vm.ref<?>>
    vm.global.store.indirect.ref %arg0, %0 : !vm.ref<?> -> !util.ptr<!vm.ref<?>>
    vm.return
  }
}

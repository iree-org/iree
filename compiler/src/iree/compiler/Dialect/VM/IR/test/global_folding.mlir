// Tests folding and canonicalization of global ops.

// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(canonicalize))" %s | FileCheck %s

// CHECK-LABEL: @global_i32_folds
vm.module @global_i32_folds {
  // CHECK: vm.global.i32 public mutable @g0 = 123 : i32
  vm.global.i32 mutable @g0 : i32
  vm.initializer {
    %c123 = vm.const.i32 123
    vm.global.store.i32 %c123, @g0 : i32
    vm.return
  }

  // CHECK: vm.global.i32 public mutable @g1 : i32
  vm.global.i32 mutable @g1 = 0 : i32
  // CHECK: vm.global.i32 public @g2 : i32
  vm.global.i32 @g2 = 0 : i32

  // CHECK: vm.global.i32 public mutable @g3 : i32
  vm.global.i32 mutable @g3 : i32
  vm.initializer {
    %c0 = vm.const.i32 0
    vm.global.store.i32 %c0, @g3 : i32
    vm.return
  }
}

// -----

// CHECK-LABEL: @global_indirect_folds
vm.module @global_indirect_folds {
  vm.global.i32 mutable @g0 : i32

  // CHECK-LABEL: @fold_load_i32
  vm.func @fold_load_i32() -> i32 {
    %0 = vm.global.address @g0 : !util.ptr<i32>
    // CHECK-NEXT: %[[VALUE:.+]] = vm.global.load.i32 @g0 : i32
    %1 = vm.global.load.indirect.i32 %0 : !util.ptr<i32> -> i32
    // CHECK-NEXT: vm.return %[[VALUE]]
    vm.return %1 : i32
  }

  // CHECK-LABEL: @fold_store_i32
  vm.func @fold_store_i32(%arg0 : i32) {
    %0 = vm.global.address @g0 : !util.ptr<i32>
    // CHECK-NEXT: vm.global.store.i32 %arg0, @g0 : i32
    vm.global.store.indirect.i32 %arg0, %0 : i32 -> !util.ptr<i32>
    vm.return
  }
}

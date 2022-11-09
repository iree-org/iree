// Tests folding and canonicalization of constant ops.

// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(cse),vm.module(canonicalize))" %s | FileCheck %s

// CHECK-LABEL: @const_i32_folds
vm.module @const_i32_folds {
  // CHECK-LABEL: @cse
  vm.func @cse() -> (i32, i32) {
    // CHECK-NEXT: %c1 = vm.const.i32 1
    // CHECK-NEXT: vm.return %c1, %c1 : i32, i32
    %0 = vm.const.i32 1
    %1 = vm.const.i32 1
    vm.return %0, %1 : i32, i32
  }

  // CHECK-LABEL: @cse_zero
  vm.func @cse_zero() -> (i32, i32) {
    // CHECK-NEXT: %zero = vm.const.i32.zero
    // CHECK-NEXT: vm.return %zero, %zero : i32, i32
    %0 = vm.const.i32 0
    %1 = vm.const.i32 0
    vm.return %0, %1 : i32, i32
  }
}

// -----

// CHECK-LABEL: @const_ref_folds
vm.module @const_ref_folds {
  // CHECK-LABEL: @cse_null
  vm.func @cse_null() -> (!vm.ref<?>, !vm.ref<?>) {
    // CHECK-NEXT: %null = vm.const.ref.zero : !vm.ref<?>
    // CHECK-NEXT: vm.return %null, %null : !vm.ref<?>, !vm.ref<?>
    %0 = vm.const.ref.zero : !vm.ref<?>
    %1 = vm.const.ref.zero : !vm.ref<?>
    vm.return %0, %1 : !vm.ref<?>, !vm.ref<?>
  }
}

// -----

// CHECK-LABEL: @const_rodata_folds
vm.module @const_rodata_folds {
  // CHECK-NEXT: vm.rodata private @r2
  vm.rodata private @r2 dense<[9, 9, 9]> : vector<3xi32>
  // CHECK-NEXT: @cse_rodata_loads
  vm.func @cse_rodata_loads() -> (!vm.buffer, !vm.buffer) {
    // CHECK-NEXT: %r2 = vm.const.ref.rodata @r2 : !vm.buffer
    // CHECK-NEXT: vm.return %r2, %r2 : !vm.buffer, !vm.buffer
    %0 = vm.const.ref.rodata @r2 : !vm.buffer
    %1 = vm.const.ref.rodata @r2 : !vm.buffer
    vm.return %0, %1 : !vm.buffer, !vm.buffer
  }
}

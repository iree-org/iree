// Tests folding and canonicalization of assignment ops.

// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(canonicalize))" %s | FileCheck %s

// CHECK-LABEL: @select_i32_folds
vm.module @select_i32_folds {
  // CHECK-LABEL: @select_i32_zero
  vm.func @select_i32_zero(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: vm.return %arg1 : i32
    %zero = vm.const.i32.zero
    %0 = vm.select.i32 %zero, %arg0, %arg1 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @select_i32_one
  vm.func @select_i32_one(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %c123 = vm.const.i32 123
    %0 = vm.select.i32 %c123, %arg0, %arg1 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @select_i32_eq
  vm.func @select_i32_eq(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: vm.return %arg1 : i32
    %0 = vm.select.i32 %arg0, %arg1, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @select_ref_folds
vm.module @select_ref_folds {
  // CHECK-LABEL: @select_ref_zero
  vm.func @select_ref_zero(%arg0 : !vm.ref<?>,
                           %arg1 : !vm.ref<?>) -> !vm.ref<?> {
    // CHECK: vm.return %arg1 : !vm.ref<?>
    %zero = vm.const.i32.zero
    %0 = vm.select.ref %zero, %arg0, %arg1 : !vm.ref<?>
    vm.return %0 : !vm.ref<?>
  }

  // CHECK-LABEL: @select_ref_one
  vm.func @select_ref_one(%arg0 : !vm.ref<?>,
                          %arg1 : !vm.ref<?>) -> !vm.ref<?> {
    // CHECK: vm.return %arg0 : !vm.ref<?>
    %c123 = vm.const.i32 123
    %0 = vm.select.ref %c123, %arg0, %arg1 : !vm.ref<?>
    vm.return %0 : !vm.ref<?>
  }

  // CHECK-LABEL: @select_ref_eq
  vm.func @select_ref_eq(%arg0 : i32,
                         %arg1 : !vm.ref<?>) -> !vm.ref<?> {
    // CHECK: vm.return %arg1 : !vm.ref<?>
    %0 = vm.select.ref %arg0, %arg1, %arg1 : !vm.ref<?>
    vm.return %0 : !vm.ref<?>
  }
}

// -----

// CHECK-LABEL: @switch_i32_folds
vm.module @switch_i32_folds {
  // CHECK-LABEL: @switch_i32_nop
  vm.func @switch_i32_nop(%arg0 : i32) -> i32 {
    %c5 = vm.const.i32 5
    // CHECK: vm.return %c5 : i32
    %0 = vm.switch.i32 %arg0[] else %c5 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @switch_i32_identical
  vm.func @switch_i32_identical(%arg0 : i32) -> i32 {
    %c100 = vm.const.i32 100
    // CHECK: vm.return %c100 : i32
    %0 = vm.switch.i32 %arg0[%c100, %c100, %c100] else %c100 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @switch_i32_constant_index
  vm.func @switch_i32_constant_index() -> (i32, i32, i32, i32) {
    %c0 = vm.const.i32 0
    %c1 = vm.const.i32 1
    %c2 = vm.const.i32 2
    %c3 = vm.const.i32 3
    %c100 = vm.const.i32 100
    %c200 = vm.const.i32 200
    %c300 = vm.const.i32 300
    %c400 = vm.const.i32 400
    // CHECK: vm.return %c100, %c200, %c300, %c400 : i32, i32, i32, i32
    %0 = vm.switch.i32 %c0[%c100, %c200, %c300] else %c400 : i32
    %1 = vm.switch.i32 %c1[%c100, %c200, %c300] else %c400 : i32
    %2 = vm.switch.i32 %c2[%c100, %c200, %c300] else %c400 : i32
    %3 = vm.switch.i32 %c3[%c100, %c200, %c300] else %c400 : i32
    vm.return %0, %1, %2, %3 : i32, i32, i32, i32
  }
}

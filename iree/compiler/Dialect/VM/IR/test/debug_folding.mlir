// Tests folding and canonicalization of debug ops.

// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(canonicalize)' %s | IreeFileCheck %s

// CHECK-LABEL: @cond_break_folds
vm.module @cond_break_folds {
  // CHECK-LABEL: @const_cond_break_true
  vm.func @const_cond_break_true(%arg0 : i32) -> i32 {
    // CHECK-NEXT: vm.break ^bb1(%arg0 : i32)
    %c1 = vm.const.i32 1 : i32
    vm.cond_break %c1, ^bb1(%arg0 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  }

  // CHECK-LABEL: @const_cond_break_false
  vm.func @const_cond_break_false(%arg0 : i32) -> i32 {
    // CHECK-NEXT: vm.return %arg0 : i32
    %zero = vm.const.i32.zero : i32
    vm.cond_break %zero, ^bb1(%arg0 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  }
}

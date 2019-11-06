// Tests folding and canonicalization of control flow ops.

// RUN: iree-opt -split-input-file \
// RUN:     -pass-pipeline='vm.module(canonicalize)' %s | \
// RUN:     FileCheck %s --dump-input=fail

// CHECK-LABEL: @cond_br_folds
vm.module @cond_br_folds {
  // CHECK-LABEL: @const_cond_br_true
  vm.func @const_cond_br_true(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK-NEXT: vm.br ^bb1(%arg0 : i32)
    %c1 = vm.const.i32 1 : i32
    vm.cond_br %c1, ^bb1(%arg0 : i32), ^bb2(%arg1 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  ^bb2(%1 : i32):
    vm.return %1 : i32
  }

  // CHECK-LABEL: @const_cond_br_false
  vm.func @const_cond_br_false(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK-NEXT: vm.br ^bb1(%arg1 : i32)
    %zero = vm.const.i32.zero : i32
    vm.cond_br %zero, ^bb1(%arg0 : i32), ^bb2(%arg1 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  ^bb2(%1 : i32):
    vm.return %1 : i32
  }

  // CHECK-LABEL: @same_target_same_args_cond_br
  vm.func @same_target_same_args_cond_br(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK-NEXT: vm.br ^bb1(%arg1 : i32)
    vm.cond_br %arg0, ^bb1(%arg1 : i32), ^bb1(%arg1 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  }

  // CHECK-LABEL: @same_target_diff_args_cond_br
  vm.func @same_target_diff_args_cond_br(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // NOTE: args differ, so cannot fold.
    // CHECK-NEXT: vm.cond_br %arg0, ^bb1(%arg1 : i32), ^bb1(%arg2 : i32)
    vm.cond_br %arg0, ^bb1(%arg1 : i32), ^bb1(%arg2 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  }

  // CHECK-LABEL: @swap_inverted_cond_br
  vm.func @swap_inverted_cond_br(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // CHECK-NEXT: vm.cond_br %arg0, ^bb2(%arg1 : i32), ^bb1(%arg0 : i32)
    %inv = vm.not.i32 %arg0 : i32
    vm.cond_br %inv, ^bb1(%arg0 : i32), ^bb2(%arg1 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  ^bb2(%1 : i32):
    vm.return %1 : i32
  }
}

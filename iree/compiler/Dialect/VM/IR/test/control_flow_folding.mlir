// Tests folding and canonicalization of control flow ops.

// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(canonicalize)' %s | IreeFileCheck %s

// CHECK-LABEL: @cond_br_folds
vm.module @cond_br_folds {
  // CHECK-LABEL: @const_cond_br_true
  vm.func @const_cond_br_true(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK-NEXT: vm.return %arg0 : i32
    %c1 = vm.const.i32 1 : i32
    vm.cond_br %c1, ^bb1(%arg0 : i32), ^bb2(%arg1 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  ^bb2(%1 : i32):
    vm.return %1 : i32
  }

  // CHECK-LABEL: @const_cond_br_false
  vm.func @const_cond_br_false(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK-NEXT: vm.return %arg1 : i32
    %zero = vm.const.i32.zero : i32
    vm.cond_br %zero, ^bb1(%arg0 : i32), ^bb2(%arg1 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  ^bb2(%1 : i32):
    vm.return %1 : i32
  }

  // CHECK-LABEL: @same_target_same_args_cond_br
  vm.func @same_target_same_args_cond_br(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK-NEXT: vm.return %arg1 : i32
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

  // TODO(benvanik): fix swapping by proper cond^1 check.
  // // DISABLED-LABEL: @swap_inverted_cond_br
  // vm.func @swap_inverted_cond_br(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
  //   // DISABLED-NEXT: vm.cond_br %arg0, ^bb2(%arg1 : i32), ^bb1(%arg0 : i32)
  //   %c1 = vm.const.i32 1 : i32
  //   %inv = vm.xor.i32 %arg0, %c1 : i32
  //   vm.cond_br %inv, ^bb1(%arg0 : i32), ^bb2(%arg1 : i32)
  // ^bb1(%0 : i32):
  //   vm.fail %0
  // ^bb2(%1 : i32):
  //   vm.return %1 : i32
  // }

  // CHECK-LABEL: @erase_unused_pure_call
  vm.func @erase_unused_pure_call(%arg0 : i32) {
    %0 = vm.call @nonvariadic_pure_func(%arg0) : (i32) -> i32
    %1 = vm.call.variadic @variadic_pure_func([%arg0]) : (i32 ...) -> i32
    // CHECK-NEXT: vm.return
    vm.return
  }
  vm.import @nonvariadic_pure_func(%arg0 : i32) -> i32 attributes {nosideeffects}
  vm.import @variadic_pure_func(%arg0 : i32 ...) -> i32 attributes {nosideeffects}

  // CHECK-LABEL: @convert_nonvariadic_to_call
  vm.func @convert_nonvariadic_to_call(%arg0 : i32) -> (i32, i32) {
    // CHECK-NEXT: vm.call @nonvariadic_func(%arg0) : (i32) -> i32
    %0 = vm.call.variadic @nonvariadic_func(%arg0) : (i32) -> i32
    // CHECK-NEXT: vm.call.variadic @variadic_func
    %1 = vm.call.variadic @variadic_func(%arg0, []) : (i32, i32 ...) -> i32
    // CHECK-NEXT: vm.return
    vm.return %0, %1 : i32, i32
  }
  vm.import @nonvariadic_func(%arg0 : i32) -> i32
  vm.import @variadic_func(%arg0 : i32, %arg1 : i32 ...) -> i32
}

// -----

// CHECK-LABEL: @cond_fail_folds
vm.module @cond_fail_folds {
  // CHECK-LABEL: @cond_fail_to_cond_br_fail
  // CHECK-SAME: %[[COND:.+]]:
  vm.func @cond_fail_to_cond_br_fail(%cond : i32) {
    // CHECK-DAG: %[[CODE2:.+]] = constant 2
    %code2 = constant 2 : i32
    // CHECK: vm.cond_br %[[COND]], ^bb2(%[[CODE2]] : i32), ^bb1
    vm.cond_fail %cond, %code2, "message"
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: vm.return
    vm.return
    // CHECK-NEXT: ^bb2(%[[STATUS:.+]]: i32):
    // CHECK-NEXT: vm.fail %[[STATUS]], "message"
  }
}

// -----

// CHECK-LABEL: @check_folds
vm.module @check_folds {
  // CHECK-LABEL: @check_eq_i32
  vm.func @check_eq_i32(%arg0 : i32, %arg1 : i32) {
    // CHECK: %[[COND:.+]] = vm.cmp.ne.i32 %arg0, %arg1 : i32
    // CHECK-NEXT: vm.cond_br %[[COND]], ^bb2({{.+}}), ^bb1
    vm.check.eq %arg0, %arg1, "expected eq" : i32
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: vm.return
    vm.return
    // CHECK-NEXT: ^bb2(%[[STATUS:.+]]: i32):
    // CHECK-NEXT: vm.fail %[[STATUS]], "expected eq"
  }

  // CHECK-LABEL: @check_nz_i32
  vm.func @check_nz_i32(%arg0 : i32) {
    // CHECK: %[[COND:.+]] = vm.cmp.nz.i32 %arg0 : i32
    // CHECK: %[[INV_COND:.+]] = vm.xor.i32 %[[COND]], %c1 : i32
    // CHECK-NEXT: vm.cond_br %[[INV_COND]], ^bb2({{.+}}), ^bb1
    vm.check.nz %arg0, "expected nz" : i32
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: vm.return
    vm.return
    // CHECK-NEXT: ^bb2(%[[STATUS:.+]]: i32):
    // CHECK-NEXT: vm.fail %[[STATUS]], "expected nz"
  }

  // CHECK-LABEL: @check_nz_ref
  vm.func @check_nz_ref(%arg0 : !vm.ref<?>) {
    // CHECK: %[[COND:.+]] = vm.cmp.nz.ref %arg0 : !vm.ref<?>
    // CHECK: %[[INV_COND:.+]] = vm.xor.i32 %[[COND]], %c1 : i32
    // CHECK-NEXT: vm.cond_br %[[INV_COND]], ^bb2({{.+}}), ^bb1
    vm.check.nz %arg0, "expected nz" : !vm.ref<?>
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: vm.return
    vm.return
    // CHECK-NEXT: ^bb2(%[[STATUS:.+]]: i32):
    // CHECK-NEXT: vm.fail %[[STATUS]], "expected nz"
  }
}

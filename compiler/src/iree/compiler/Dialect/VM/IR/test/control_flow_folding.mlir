// Tests folding and canonicalization of control flow ops.

// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(canonicalize))" %s | FileCheck %s

// CHECK-LABEL: @cond_br_folds
vm.module @cond_br_folds {
  // CHECK-LABEL: @const_cond_br_true
  vm.func @const_cond_br_true(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK-NEXT: vm.return %arg0 : i32
    %c1 = vm.const.i32 1
    vm.cond_br %c1, ^bb1(%arg0 : i32), ^bb2(%arg1 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  ^bb2(%1 : i32):
    vm.return %1 : i32
  }

  // CHECK-LABEL: @const_cond_br_false
  vm.func @const_cond_br_false(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK-NEXT: vm.return %arg1 : i32
    %zero = vm.const.i32.zero
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
  //   %c1 = vm.const.i32 1
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
  vm.import private @nonvariadic_pure_func(%arg0 : i32) -> i32 attributes {nosideeffects}
  vm.import private @variadic_pure_func(%arg0 : i32 ...) -> i32 attributes {nosideeffects}

  // CHECK-LABEL: @convert_nonvariadic_to_call
  vm.func @convert_nonvariadic_to_call(%arg0 : i32) -> (i32, i32) {
    // CHECK-NEXT: vm.call @nonvariadic_func(%arg0) : (i32) -> i32
    %0 = vm.call.variadic @nonvariadic_func(%arg0) : (i32) -> i32
    // CHECK-NEXT: vm.call.variadic @variadic_func
    %1 = vm.call.variadic @variadic_func(%arg0, []) : (i32, i32 ...) -> i32
    // CHECK-NEXT: vm.return
    vm.return %0, %1 : i32, i32
  }
  vm.import private @nonvariadic_func(%arg0 : i32) -> i32
  vm.import private @variadic_func(%arg0 : i32, %arg1 : i32 ...) -> i32
}

// -----

// CHECK-LABEL: @cond_fail_folds
vm.module @cond_fail_folds {
  // CHECK-LABEL: @cond_fail_to_cond_br_fail
  // CHECK-SAME: %[[COND:.+]]:
  vm.func @cond_fail_to_cond_br_fail(%cond : i32) {
    // CHECK-DAG: %[[CODE2:.+]] = arith.constant 2
    %code2 = arith.constant 2 : i32
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

  // CHECK-LABEL: @check_nearly_eq_f32
  vm.func @check_nearly_eq_f32(%arg0 : f32, %arg1 : f32) {
    // CHECK-NEXT:   %zero = vm.const.f32.zero
    // CHECK-NEXT:   %c1 = vm.const.i32 1
    // CHECK-NEXT:   [[THRESHOLD:%.+]] = vm.const.i32 100
    // CHECK-NEXT:   %c9 = vm.const.i32 9
    // CHECK-NEXT:   %0 = vm.cmp.lt.f32.o %arg0, %zero : f32
    // CHECK-NEXT:   %1 = vm.xor.i32 %0, %c1 : i32
    // CHECK-NEXT:   %2 = vm.cmp.lt.f32.o %arg1, %zero : f32
    // CHECK-NEXT:   %3 = vm.xor.i32 %2, %c1 : i32
    // CHECK-NEXT:   %ne = vm.cmp.ne.i32 %1, %3 : i32
    // CHECK-NEXT:   vm.cond_br %ne, ^bb1, ^bb2
    // CHECK-NEXT: ^bb1:  // pred: ^bb0
    // CHECK-NEXT:   %4 = vm.cmp.eq.f32.o %arg0, %arg1 : f32
    // CHECK-NEXT:   vm.br ^bb3(%4 : i32)
    // CHECK-NEXT: ^bb2:  // pred: ^bb0
    // CHECK-NEXT:   %5 = vm.bitcast.f32.i32 %arg0 : f32 -> i32
    // CHECK-NEXT:   %6 = vm.bitcast.f32.i32 %arg1 : f32 -> i32
    // CHECK-NEXT:   %7 = vm.sub.i32 %5, %6 : i32
    // CHECK-NEXT:   %8 = vm.abs.i32 %7 : i32
    // CHECK-NEXT:   %slt = vm.cmp.lt.i32.s %8, [[THRESHOLD]] : i32
    // CHECK-NEXT:   vm.br ^bb3(%slt : i32)
    // CHECK-NEXT: ^bb3(%9: i32):  // 2 preds: ^bb1, ^bb2
    // CHECK-NEXT:   %10 = vm.xor.i32 %9, %c1 : i32
    // CHECK-NEXT:   vm.cond_br %10, ^bb5(%c9 : i32), ^bb4
    vm.check.nearly_eq %arg0, %arg1, "expected nearly eq" : f32
    // CHECK-NEXT: ^bb4:
    // CHECK-NEXT:   vm.return
    vm.return
    // CHECK-NEXT: ^bb5(%[[STATUS:.+]]: i32):
    // CHECK-NEXT:   vm.fail %[[STATUS]], "expected nearly eq"
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

// -----

// CHECK-LABEL: @check_imports
vm.module @check_imports {
  vm.import private @required_import_fn(%arg0 : i32) -> i32
  vm.import private optional @optional_import_fn(%arg0 : i32) -> i32
  vm.func @call_fn() -> (i32, i32) {
    // CHECK-NOT: vm.import.resolved @required_import_fn
    // CHECK-DAG: %[[HAS_STRONG:.+]] = vm.const.i32 1
    %has_required_import_fn = vm.import.resolved @required_import_fn : i32
    // CHECK-DAG: %[[HAS_WEAK:.+]] = vm.import.resolved @optional_import_fn : i32
    %has_optional_import_fn = vm.import.resolved @optional_import_fn : i32
    vm.return %has_required_import_fn, %has_optional_import_fn : i32, i32
  }
}

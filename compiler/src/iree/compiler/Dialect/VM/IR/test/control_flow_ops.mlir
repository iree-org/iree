// RUN: iree-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: @branch_empty
vm.module @my_module {
  vm.func @branch_empty() {
    // CHECK: vm.br ^bb1
    vm.br ^bb1
  ^bb1:
    vm.return
  }
}

// -----

// CHECK-LABEL: @branch_args
vm.module @my_module {
  vm.func @branch_args(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: vm.br ^bb1(%arg0, %arg1 : i32, i32)
    vm.br ^bb1(%arg0, %arg1 : i32, i32)
  ^bb1(%0 : i32, %1 : i32):
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @cond_branch_empty
vm.module @my_module {
  vm.func @cond_branch_empty(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // CHECK: vm.cond_br %arg0, ^bb1, ^bb2
    vm.cond_br %arg0, ^bb1, ^bb2
  ^bb1:
    vm.return %arg1 : i32
  ^bb2:
    vm.return %arg2 : i32
  }
}

// -----

// CHECK-LABEL: @cond_branch_args
vm.module @my_module {
  vm.func @cond_branch_args(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // CHECK: vm.cond_br %arg0, ^bb1(%arg1 : i32), ^bb2(%arg2 : i32)
    vm.cond_br %arg0, ^bb1(%arg1 : i32), ^bb2(%arg2 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  ^bb2(%1 : i32):
    vm.return %1 : i32
  }
}

// -----

// CHECK-LABEL: @call_fn
vm.module @my_module {
  vm.import private @import_fn(%arg0 : i32) -> i32
  vm.func @call_fn(%arg0 : i32) -> i32 {
    // CHECK: %0 = vm.call @import_fn(%arg0) : (i32) -> i32
    %0 = vm.call @import_fn(%arg0) : (i32) -> i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @call_variadic_but_not_really
vm.module @my_module {
  vm.import private @import_fn(%arg0 : i32) -> i32
  vm.func @call_variadic_but_not_really(%arg0 : i32) -> i32 {
    // CHECK: %0 = vm.call.variadic @import_fn(%arg0) : (i32) -> i32
    %0 = vm.call.variadic @import_fn(%arg0) : (i32) -> i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @call_variadic_empty
vm.module @my_module {
  vm.import private @import_fn(%arg0 : i32, %arg1 : !vm.ref<!hal.buffer> ...) -> i32
  vm.func @call_variadic_empty(%arg0 : i32) -> i32 {
    // CHECK: %0 = vm.call.variadic @import_fn(%arg0, []) : (i32, !vm.ref<!hal.buffer> ...) -> i32
    %0 = vm.call.variadic @import_fn(%arg0, []) : (i32, !vm.ref<!hal.buffer> ...) -> i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @call_variadic
vm.module @my_module {
  vm.import private @import_fn(%arg0 : i32, %arg1 : !vm.ref<!hal.buffer> ...) -> i32
  vm.func @call_variadic(%arg0 : i32, %arg1 : !vm.ref<!hal.buffer>) -> i32 {
    // CHECK: %0 = vm.call.variadic @import_fn(%arg0, [%arg1, %arg1]) : (i32, !vm.ref<!hal.buffer> ...) -> i32
    %0 = vm.call.variadic @import_fn(%arg0, [%arg1, %arg1]) : (i32, !vm.ref<!hal.buffer> ...) -> i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @call_variadic_multiple
vm.module @my_module {
  vm.import private @import_fn(%arg0 : i32, %arg1 : !vm.ref<!hal.buffer> ...) -> i32
  vm.func @call_variadic_multiple(%arg0 : i32, %arg1 : !vm.ref<!hal.buffer>) -> i32 {
    // CHECK: %0 = vm.call.variadic @import_fn(%arg0, [%arg1, %arg1], [%arg1]) : (i32, !vm.ref<!hal.buffer> ..., !vm.ref<!hal.buffer> ...) -> i32
    %0 = vm.call.variadic @import_fn(%arg0, [%arg1, %arg1], [%arg1]) : (i32, !vm.ref<!hal.buffer> ..., !vm.ref<!hal.buffer> ...) -> i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @call_variadic_no_results
vm.module @my_module {
  vm.import private @import_fn(%arg0 : i32, %arg1 : !vm.ref<!hal.buffer> ...)
  vm.func @call_variadic_no_results(%arg0 : i32, %arg1 : !vm.ref<!hal.buffer>) {
    // CHECK: vm.call.variadic @import_fn(%arg0, [%arg1, %arg1], [%arg1]) : (i32, !vm.ref<!hal.buffer> ..., !vm.ref<!hal.buffer> ...)
    vm.call.variadic @import_fn(%arg0, [%arg1, %arg1], [%arg1]) : (i32, !vm.ref<!hal.buffer> ..., !vm.ref<!hal.buffer> ...)
    vm.return
  }
}

// -----

// CHECK-LABEL: @call_variadic_tuples
vm.module @my_module {
  vm.import private @import_fn(%arg0 : tuple<i32, i32, i32> ...)
  vm.func @call_variadic_tuples(%arg0 : i32, %arg1 : i32) {
    // CHECK: vm.call.variadic @import_fn([(%arg0, %arg0, %arg0), (%arg1, %arg1, %arg1)]) : (tuple<i32, i32, i32> ...)
    vm.call.variadic @import_fn([(%arg0, %arg0, %arg0), (%arg1, %arg1, %arg1)]) : (tuple<i32, i32, i32> ...)
    vm.return
  }
}

// -----

// CHECK-LABEL: @return_empty
vm.module @my_module {
  vm.func @return_empty() {
    // CHECK: vm.return
    vm.return
  }
}

// -----

// CHECK-LABEL: @return_args
vm.module @my_module {
  vm.func @return_args(%arg0 : i32, %arg1 : i32) -> (i32, i32) {
    // CHECK: vm.return %arg0, %arg1 : i32, i32
    vm.return %arg0, %arg1 : i32, i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @fail
  vm.func @fail() {
    // CHECK-DAG: %[[CODE1:.+]] = arith.constant 1
    %code1 = arith.constant 1 : i32
    // CHECK: vm.fail %[[CODE1]]
    vm.fail %code1
  }
  // CHECK-LABEL: @fail_message
  vm.func @fail_message() {
    // CHECK-DAG: %[[CODE2:.+]] = arith.constant 2
    %code2 = arith.constant 2 : i32
    // CHECK: vm.fail %[[CODE2]], "message"
    vm.fail %code2, "message"
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @cond_fail
  // CHECK-SAME: %[[COND:.+]]:
  vm.func @cond_fail(%cond : i32) {
    // CHECK-DAG: %[[CODE1:.+]] = arith.constant 1
    %code1 = arith.constant 1 : i32
    // CHECK: vm.cond_fail %[[COND]], %[[CODE1]]
    vm.cond_fail %cond, %code1
    vm.return
  }
  // CHECK-LABEL: @cond_fail_message
  // CHECK-SAME: %[[COND:.+]]:
  vm.func @cond_fail_message(%cond : i32) {
    // CHECK-DAG: %[[CODE2:.+]] = arith.constant 2
    %code2 = arith.constant 2 : i32
    // CHECK: vm.cond_fail %[[COND]], %[[CODE2]], "message"
    vm.cond_fail %cond, %code2, "message"
    vm.return
  }
  // CHECK-LABEL: @cond_fail_no_condition
  vm.func @cond_fail_no_condition() {
    // CHECK-DAG: %[[CODE3:.+]] = arith.constant 3
    %code3 = arith.constant 3 : i32
    // CHECK: vm.cond_fail %[[CODE3]]
    vm.cond_fail %code3
    vm.return
  }
  // CHECK-LABEL: @cond_fail_no_condition_with_message
  vm.func @cond_fail_no_condition_with_message() {
    // CHECK-DAG: %[[CODE4:.+]] = arith.constant 4
    %code4 = arith.constant 4 : i32
    // CHECK: vm.cond_fail %[[CODE4]]
    vm.cond_fail %code4, "message"
    vm.return
  }
}

// -----

// CHECK-LABEL: @yield
vm.module @my_module {
  vm.func @yield() {
    // CHECK: vm.yield ^bb1
    vm.yield ^bb1
  ^bb1:
    vm.return
  }
}

// -----

vm.module @my_module {
  // CHECK: vm.import private optional @optional_import_fn(%arg0 : i32) -> i32
  vm.import private optional @optional_import_fn(%arg0 : i32) -> i32
  vm.func @call_fn() -> i32 {
    // CHECK: %has_optional_import_fn = vm.import.resolved @optional_import_fn : i32
    %has_optional_import_fn = vm.import.resolved @optional_import_fn : i32
    vm.return %has_optional_import_fn : i32
  }
}

// Tests printing and parsing of control flow ops.

// RUN: iree-opt -split-input-file %s | \
// RUN:     FileCheck %s --dump-input=fail

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
  vm.func @import_fn(%arg0 : i32) -> i32
  vm.func @call_fn(%arg0 : i32) -> i32 {
    // CHECK: %0 = vm.call @import_fn(%arg0) : (i32) -> i32
    %0 = vm.call @import_fn(%arg0) : (i32) -> i32
    vm.return %0 : i32
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

// CHECK-LABEL: @yield
vm.module @my_module {
  vm.func @yield() {
    // CHECK: vm.yield
    vm.yield
    vm.return
  }
}

// RUN: iree-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: @trace_args
vm.module @my_module {
  vm.func @trace_args(%arg0 : i32, %arg1 : i32) {
    // CHECK: vm.trace "event"(%arg0, %arg1) : i32, i32
    vm.trace "event"(%arg0, %arg1) : i32, i32
    vm.return
  }
}

// -----

// CHECK-LABEL: @print_args
vm.module @my_module {
  vm.func @print_args(%arg0 : i32, %arg1 : i32) {
    // CHECK: vm.print "message"(%arg0, %arg1) : i32, i32
    vm.print "message"(%arg0, %arg1) : i32, i32
    vm.return
  }
}

// -----

// CHECK-LABEL: @break_empty
vm.module @my_module {
  vm.func @break_empty() {
    // CHECK: vm.break ^bb1
    vm.break ^bb1
  ^bb1:
    vm.return
  }
}

// -----

// CHECK-LABEL: @break_args
vm.module @my_module {
  vm.func @break_args(%arg0 : i32) -> i32 {
    // CHECK: vm.break ^bb1(%arg0 : i32)
    vm.break ^bb1(%arg0 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @cond_break_empty
vm.module @my_module {
  vm.func @cond_break_empty(%arg0 : i32) {
    // CHECK: vm.cond_break %arg0, ^bb1
    vm.cond_break %arg0, ^bb1
  ^bb1:
    vm.return
  }
}

// -----

// CHECK-LABEL: @break_args
vm.module @my_module {
  vm.func @break_args(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: vm.cond_break %arg0, ^bb1(%arg1 : i32)
    vm.cond_break %arg0, ^bb1(%arg1 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  }
}

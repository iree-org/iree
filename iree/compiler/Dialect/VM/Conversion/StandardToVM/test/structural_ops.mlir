// RUN: iree-opt -split-input-file -pass-pipeline='test-iree-convert-std-to-vm' %s | FileCheck %s

// -----
// Checks literal specifics of structural transforms (more verbose checks
// than usual since the conversion code is manual).
// CHECK-LABEL: @t001_module_all_options
module @t001_module_all_options {

// CHECK: vm.module public @my_module {
module @my_module {
  // CHECK: vm.func private @my_fn(%[[ARG0:[a-zA-Z0-9$._-]+]]: i32) -> i32
  func @my_fn(%arg0: i32) -> (i32) {
    // CHECK: vm.return %[[ARG0]] : i32
    return %arg0 : i32
  }
}

}

// -----
// CHECK-LABEL: @t002_no_args_results
module @t002_no_args_results {

module @my_module {
  // CHECK: vm.func private @my_fn() {
  func @my_fn() -> () {
    // CHECK: vm.return
    return
  }
}

}

// -----
// CHECK-LABEL: @t003_unnamed_module
module @t003_unnamed_module {

// CHECK: vm.module public @module {
module {
}

}

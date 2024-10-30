// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-drop-unused-calls))" %s | FileCheck %s

// CHECK-LABEL: @drop_calls
vm.module public @drop_calls {
  // CHECK: vm.func @fn
  vm.func @fn(%arg0 : i32) {
    // CHECK-NOT: vm.call @nonvariadic_pure_func
    %0 = vm.call @nonvariadic_pure_func(%arg0) : (i32) -> i32
    // CHECK-NOT: vm.call.variadic @variadic_pure_func
    %1 = vm.call.variadic @variadic_pure_func([%arg0]) : (i32 ...) -> i32
    // CHECK-NEXT: vm.return
    vm.return
  }
  vm.import private @nonvariadic_pure_func(%arg0 : i32) -> i32 attributes {nosideeffects}
  vm.import private @variadic_pure_func(%arg0 : i32 ...) -> i32 attributes {nosideeffects}
}

// -----

// CHECK-LABEL: @drop_call_trees
vm.module public @drop_call_trees {
  // CHECK: vm.func @fn
  vm.func @fn(%arg0 : i32) {
    // CHECK: vm.call @impure_func
    %0 = vm.call @impure_func(%arg0) : (i32) -> i32
    // CHECK-NOT: vm.call @pure_func_a
    %1 = vm.call @pure_func_a(%0) : (i32) -> i32
    // CHECK-NOT: vm.call @pure_func_b
    %2 = vm.call @pure_func_b(%1) : (i32) -> i32
    // CHECK-NEXT: vm.return
    vm.return
  }
  vm.import private @impure_func(%arg0 : i32) -> i32
  vm.import private @pure_func_a(%arg0 : i32) -> i32 attributes {nosideeffects}
  vm.import private @pure_func_b(%arg0 : i32) -> i32 attributes {nosideeffects}
}

// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s


vm.module @my_module {
  // CHECK-LABEL: @my_module_const_i64_zero
  vm.func @const_i64_zero() -> i64 {
    // CHECK: %[[ZERO:.+]] = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %zero = vm.const.i64.zero
    vm.return %zero : i64
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_const_i64
  vm.func @const_i64() {
    // CHECK-NEXT: %0 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %0 = vm.const.i64 0
    // CHECK-NEXT: %1 = "emitc.constant"() <{value = 2 : i64}> : () -> i64
    %1 = vm.const.i64 2
    // CHECK-NEXT: %2 = "emitc.constant"() <{value = -2 : i64}> : () -> i64
    %2 = vm.const.i64 -2
    vm.return
  }
}

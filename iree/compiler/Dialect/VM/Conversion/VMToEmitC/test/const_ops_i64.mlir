// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-convert-vm-to-emitc)' %s | IreeFileCheck %s


vm.module @my_module {
  // CHECK-LABEL: vm.func @const_i64_zero
  vm.func @const_i64_zero() -> i64 {
    // CHECK: %zero = "emitc.const"() {value = 0 : i64} : () -> i64
    %zero = vm.const.i64.zero : i64
    vm.return %zero : i64
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: vm.func @const_i64
  vm.func @const_i64() {
    // CHECK-NEXT: %0 = "emitc.const"() {value = 0 : i64} : () -> i64
    %0 = vm.const.i64 0 : i64
    // CHECK-NEXT: %1 = "emitc.const"() {value = 2 : i64} : () -> i64
    %1 = vm.const.i64 2 : i64
    // CHECK-NEXT: %2 = "emitc.const"() {value = -2 : i64} : () -> i64
    %2 = vm.const.i64 -2 : i64
    vm.return
  }
}

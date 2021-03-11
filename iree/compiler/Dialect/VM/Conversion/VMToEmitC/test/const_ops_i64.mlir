// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-convert-vm-to-emitc)' %s | IreeFileCheck %s

// CHECK: vm.module @module {
vm.module @module {
  // CHECK-LABEL: vm.func @const_i64
  vm.func @const_i64() {
    // CHECK-NEXT: %0 = "emitc.const"() {value = 0 : i64} : () -> i64
    %0 = vm.const.i64 0 : i64
    // CHECK-NEXT: %1 = "emitc.const"() {value = 2 : i64} : () -> i64
    %1 = vm.const.i64 2 : i64
    // CHECK-NEXT: %2 = "emitc.const"() {value = -2 : i64} : () -> i64
    %2 = vm.const.i64 -2 : i64
    // CHECK-NEXT: vm.return
    vm.return
  }
}

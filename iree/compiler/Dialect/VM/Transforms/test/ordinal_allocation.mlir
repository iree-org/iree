// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-vm-ordinal-allocation)' %s | IreeFileCheck %s

// CHECK-LABEL: @global_address_propagation
vm.module @global_address_propagation {
  // CHECK-DAG: vm.global.i32 @g0 mutable : i32 attributes {ordinal = 0 : i32}
  vm.global.i32 @g0 mutable : i32
  // CHECK-DAG: vm.global.i32 @g1 mutable : i32 attributes {ordinal = 4 : i32}
  vm.global.i32 @g1 mutable : i32

  // CHECK-NEXT: @main
  vm.func @main() -> i32 {
    // CHECK-NEXT: %[[G0_ADDR:.+]] = vm.const.i32 0
    %0 = vm.global.address @g0 : !iree.ptr<i32>
    // CHECK-NEXT: vm.global.load.indirect.i32 %[[G0_ADDR]]
    %1 = vm.global.load.indirect.i32 %0 : !iree.ptr<i32> -> i32
    // CHECK-NEXT: %[[G1_ADDR:.+]] = vm.const.i32 4
    %2 = vm.global.address @g1 : !iree.ptr<i32>
    // CHECK-NEXT: vm.global.load.indirect.i32 %[[G1_ADDR]]
    %3 = vm.global.load.indirect.i32 %2 : !iree.ptr<i32> -> i32
    vm.return %1, %3 : i32, i32
  }
}

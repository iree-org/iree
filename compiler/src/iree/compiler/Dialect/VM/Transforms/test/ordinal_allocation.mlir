// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation))" %s | FileCheck %s
// check the parser for vm.module.ordinal_counts
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation))" %s | iree-opt | FileCheck %s

// CHECK-LABEL: @global_address_propagation
  // CHECK-SAME: attributes {ordinal_counts = #vm.ordinal_counts<
  // CHECK-SAME: import_funcs = 0,
  // CHECK-SAME: export_funcs = 0,
  // CHECK-SAME: internal_funcs = 1,
  // CHECK-SAME: global_bytes = 8,
  // CHECK-SAME: global_refs = 0,
  // CHECK-SAME: rodatas = 0,
  // CHECK-SAME: rwdatas = 0
  // CHECK-SAME: >}
vm.module @global_address_propagation {
  // CHECK-DAG: vm.global.i32 public mutable @g0 {ordinal = 0 : i32} : i32
  vm.global.i32 mutable @g0 : i32
  // CHECK-DAG: vm.global.i32 public mutable @g1 {ordinal = 4 : i32} : i32
  vm.global.i32 mutable @g1 : i32

  // CHECK-NEXT: @main
  vm.func @main() -> i32 {
    // CHECK-NEXT: %[[G0_ADDR:.+]] = vm.const.i32 0
    %0 = vm.global.address @g0 : !util.ptr<i32>
    // CHECK-NEXT: vm.global.load.indirect.i32 %[[G0_ADDR]]
    %1 = vm.global.load.indirect.i32 %0 : !util.ptr<i32> -> i32
    // CHECK-NEXT: %[[G1_ADDR:.+]] = vm.const.i32 4
    %2 = vm.global.address @g1 : !util.ptr<i32>
    // CHECK-NEXT: vm.global.load.indirect.i32 %[[G1_ADDR]]
    %3 = vm.global.load.indirect.i32 %2 : !util.ptr<i32> -> i32
    vm.return %1, %3 : i32, i32
  }
}

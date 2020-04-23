// RUN: iree-opt -split-input-file -iree-vm-conversion %s | IreeFileCheck %s

// CHECK-LABEL: @unreachable_block
module @unreachable_block {
module {
  // CHECK: vm.func @my_fn
  func @my_fn() {
    // CHECK-NEXT: %[[CODE:.+]] = vm.const.i32 2
    // CHECK-NEXT: vm.fail %[[CODE]], "unreachable location reached"
    iree.unreachable
  }
}
}

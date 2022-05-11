// RUN: iree-opt --split-input-file --iree-vm-conversion %s | FileCheck %s

// CHECK-LABEL: @unreachable_block
module @unreachable_block {
module {
  // CHECK: vm.func private @my_fn
  func.func @my_fn() {
    // CHECK-NEXT: %[[CODE:.+]] = vm.const.i32 2
    // CHECK-NEXT: vm.fail %[[CODE]], "nope!"
    util.unreachable "nope!"
  }
}
}

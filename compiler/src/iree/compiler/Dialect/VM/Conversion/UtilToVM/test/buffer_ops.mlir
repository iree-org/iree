// RUN: iree-opt --split-input-file --iree-vm-conversion %s | FileCheck %s

// CHECK-LABEL: @buffer_constant
module @buffer_constant {
module {
  // CHECK: vm.func private @my_fn
  func.func @my_fn() {
    // CHECK-NEXT: = vm.rodata.inline : !vm.buffer = dense<[1, 2, 3]> : tensor<3xi32>
    %0 = util.buffer.constant : !util.buffer = dense<[1, 2, 3]> : tensor<3xi32>
    return
  }
}
}

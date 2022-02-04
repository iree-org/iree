// RUN: iree-opt -split-input-file -iree-vm-conversion %s | FileCheck %s

// CHECK-LABEL: @byte_buffer_constant
module @byte_buffer_constant {
module {
  // CHECK: vm.func private @my_fn
  func @my_fn() {
    // CHECK-NEXT: = vm.rodata.inline : !vm.buffer = dense<[1, 2, 3]> : tensor<3xi32>
    %0 = util.byte_buffer.constant : !util.byte_buffer = dense<[1, 2, 3]> : tensor<3xi32>
    return
  }
}
}

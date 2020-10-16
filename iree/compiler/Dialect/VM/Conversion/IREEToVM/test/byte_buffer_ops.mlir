// RUN: iree-opt -split-input-file -iree-vm-conversion %s | IreeFileCheck %s

// CHECK-LABEL: @byte_buffer_constant
module @byte_buffer_constant {
module {
  // CHECK: vm.func @my_fn
  func @my_fn() {
    // CHECK-NEXT: = vm.rodata.inline : !vm.ref<!iree.byte_buffer> = dense<[1, 2, 3]> : tensor<3xi32>
    %0 = iree.byte_buffer.constant : !iree.byte_buffer = dense<[1, 2, 3]> : tensor<3xi32>
    return
  }
}
}

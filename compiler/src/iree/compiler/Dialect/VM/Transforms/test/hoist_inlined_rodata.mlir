// RUN: iree-opt -split-input-file -iree-vm-hoist-inlined-rodata %s | FileCheck %s

vm.module @module {
  // CHECK: vm.rodata private @fn_const dense<[1, 2, 3]> : tensor<3xi32>
  // CHECK-LABEL: vm.func @fn
  vm.func @fn() {
    // CHECK: = vm.const.ref.rodata @fn_const : !vm.buffer
    %0 = vm.rodata.inline : !vm.buffer = dense<[1, 2, 3]> : tensor<3xi32>
    vm.return
  }
}

// -----

vm.module @module {
  // CHECK: vm.rodata private @_const dense<[1, 2, 3]> : tensor<3xi32>
  // CHECK-LABEL: vm.initializer
  vm.initializer {
    // CHECK: = vm.const.ref.rodata @_const : !vm.buffer
    %0 = vm.rodata.inline : !vm.buffer = dense<[1, 2, 3]> : tensor<3xi32>
    vm.return
  }
}

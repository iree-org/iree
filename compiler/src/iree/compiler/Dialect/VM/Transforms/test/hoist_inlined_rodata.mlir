// RUN: iree-opt --split-input-file --iree-vm-hoist-inlined-rodata %s | FileCheck %s

vm.module @module {
  // Here to check the symbol deduplication logic:
  vm.func @name() { vm.return }
  // CHECK: vm.rodata private @name_0 {alignment = 64 : i64, mime_type = "text/plain"} dense<[1, 2, 3]> : tensor<3xi32>
  // CHECK-LABEL: vm.func @fn
  vm.func @fn() {
    // CHECK: = vm.const.ref.rodata @name_0 : !vm.buffer
    %0 = vm.rodata.inline "name" {alignment = 64 : i64, mime_type = "text/plain"} : !vm.buffer = dense<[1, 2, 3]> : tensor<3xi32>
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

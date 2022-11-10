// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(canonicalize))" %s | FileCheck %s

// CHECK-LABEL: @empty_initializer
vm.module @empty_initializer {
  // CHECK-NOT: vm.initializer
  vm.initializer {
    vm.return
  }
}

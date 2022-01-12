// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(canonicalize)' %s | FileCheck %s

// CHECK-LABEL: @empty_initializer
vm.module @empty_initializer {
  // CHECK-NOT: vm.initializer
  vm.initializer {
    vm.return
  }
}

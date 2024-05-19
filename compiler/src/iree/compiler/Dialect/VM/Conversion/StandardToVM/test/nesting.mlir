// RUN: iree-opt --split-input-file --iree-vm-conversion --cse --iree-vm-target-index-bits=32 %s | FileCheck %s

// Note that checks are ambiguous between "module" and "vm.module" so we rely
// on vm.module printing as `vm.module public @foo`

// CHECK-LABEL: module @outerBuiltinModule
module @outerBuiltinModule {
  // CHECK-NEXT: module @innerBuiltinModule attributes {vm.toplevel}
  module @innerBuiltinModule attributes {vm.toplevel} {
    // CHECK-NEXT: vm.module public @outerVmModule
    module @outerVmModule {
      // CHECK-NEXT: vm.module public @deeplyNested
      module @deeplyNested {
        // CHECK: vm.func private @foo
        func.func @foo() {
          return
        }
      }
    }
  }
}

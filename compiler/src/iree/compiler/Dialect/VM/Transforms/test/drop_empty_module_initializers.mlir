// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-drop-empty-module-initializers))" %s | FileCheck %s

// Tests an empty module is ignored.

// CHECK: vm.module public @module_empty {
// CHECK-NEXT: }
vm.module public @module_empty {
}

// -----

// Tests that an empty initializer is removed.

// CHECK-LABEL: @init_empty
vm.module @init_empty {
  // CHECK-NOT: vm.export @__init
  vm.export @__init
  // CHECK-NOT: vm.func private @__init()
  vm.func private @__init() {
    vm.return
  }
}

// -----

// Tests that a non-empty initializer is not removed.

// CHECK-LABEL: @init_nonempty
vm.module @init_nonempty {
  // CHECK: vm.global.ref private mutable @g0 : !vm.ref<?>
  vm.global.ref private mutable @g0 : !vm.ref<?>

  // CHECK: vm.export @__init
  vm.export @__init
  // CHECK-NEXT: vm.func private @__init()
  vm.func private @__init() {
    // CHECK-NEXT: %[[NULL:.+]] = vm.const.ref.zero
    %null = vm.const.ref.zero : !vm.ref<?>
    // CHECK-NEXT: vm.global.store.ref %[[NULL]], @g0
    vm.global.store.ref %null, @g0 : !vm.ref<?>
    vm.return
  }
}

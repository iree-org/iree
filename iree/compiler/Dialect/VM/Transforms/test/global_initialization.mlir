// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-vm-global-initialization)' %s | FileCheck %s

// CHECK: vm.module public @initEmpty {
// CHECK: }
vm.module @initEmpty {
}

// -----

// CHECK-LABEL: @initI32
vm.module @initI32 {
  // CHECK: vm.global.i32 private @g0
  vm.global.i32 private @g0 : i32 = 0 : i32

  // CHECK: vm.global.i32 private mutable @g1 : i32
  vm.global.i32 private mutable @g1 = 123 : i32

  // CHECK: vm.global.i32 private mutable @g2 : i32
  vm.global.i32 private @g2 = 123 : i32

  // CHECK: vm.func private @__init() {
  // CHECK-NEXT:   %c123 = vm.const.i32 123 : i32
  // CHECK-NEXT:   vm.global.store.i32 %c123, @g1
  // CHECK-NEXT:   %c123_0 = vm.const.i32 123 : i32
  // CHECK-NEXT:   vm.global.store.i32 %c123_0, @g2
  // CHECK-NEXT:   vm.return
  // CHECK-NEXT: }
}

// -----

// CHECK-LABEL: @initRef
vm.module @initRef {
  // CHECK: vm.global.ref private mutable @g0 : !vm.ref<?>
  vm.global.ref private mutable @g0 : !vm.ref<?>

  // CHECK: vm.global.ref private mutable @g1 : !vm.ref<?>
  vm.global.ref private mutable @g1 : !vm.ref<?>

  // CHECK: vm.global.ref private @g2 : !vm.ref<?>
  vm.global.ref private @g2 : !vm.ref<?>

  // CHECK-NOT: vm.func private @__init()
}

// -----

// CHECK-LABEL: @initializers
vm.module @initializers {
  // CHECK: vm.global.i32 private mutable @g0 : i32
  vm.global.i32 private @g0 : i32
  // CHECK-NOT: vm.initializer
  vm.initializer {
    %c123 = vm.const.i32 123 : i32
    vm.global.store.i32 %c123, @g0 : i32
    vm.return
  }

  // CHECK: vm.global.ref private mutable @g1 : !vm.ref<?>
  vm.global.ref private mutable @g1 : !vm.ref<?>
  // CHECK-NOT: vm.initializer
  vm.initializer {
    %null = vm.const.ref.zero : !vm.ref<?>
    vm.global.store.ref %null, @g1 : !vm.ref<?>
    vm.return
  }

  // CHECK: vm.global.ref private mutable @g2 : !vm.ref<?>
  vm.global.ref private mutable @g2 : !vm.ref<?>
  // CHECK-NOT: vm.initializer
  vm.initializer {
    %g1 = vm.global.load.ref @g1 : !vm.ref<?>
    vm.global.store.ref %g1, @g2 : !vm.ref<?>
    vm.return
  }

  //      CHECK: vm.func private @__init() {
  // CHECK-NEXT:   %c123 = vm.const.i32 123 : i32
  // CHECK-NEXT:   vm.global.store.i32 %c123, @g0 : i32
  // CHECK-NEXT:   %null = vm.const.ref.zero : !vm.ref<?>
  // CHECK-NEXT:   vm.global.store.ref %null, @g1 : !vm.ref<?>
  // CHECK-NEXT:   %g1 = vm.global.load.ref @g1 : !vm.ref<?>
  // CHECK-NEXT:   vm.global.store.ref %g1, @g2 : !vm.ref<?>
  // CHECK-NEXT:   vm.return
  // CHECK-NEXT: }
}

// -----

// CHECK-LABEL: @unused_globals
vm.module @unused_globals {
  // CHECK: vm.global.i32 private mutable @used
  vm.global.i32 private @used : i32 = 1 : i32
  // CHECK-NOT: vm.global.i32 private @unused
  vm.global.i32 private @unused : i32 = 2 : i32
  vm.func @foo() {
    %0 = vm.global.load.i32 @used : i32
    vm.return
  }
}

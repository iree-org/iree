// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-vm-global-initialization)' %s | IreeFileCheck %s

// CHECK: vm.module public @initEmpty {
// CHECK: }
vm.module @initEmpty {
}

// -----

// CHECK-LABEL: @initI32
vm.module @initI32 {
  // CHECK: vm.global.i32 public mutable @g0 : i32
  vm.global.i32 mutable @g0 initializer(@g0init) : i32
  vm.func @g0init() -> i32 {
    %c123 = vm.const.i32 123 : i32
    vm.return %c123 : i32
  }

  // CHECK: vm.global.i32 public mutable @g1 : i32
  vm.global.i32 mutable @g1 = 123 : i32

  // CHECK: vm.global.i32 public mutable @g2 : i32
  vm.global.i32 @g2 = 123 : i32

  // CHECK: vm.func private @__init() {
  // CHECK-NEXT:   %0 = vm.call @g0init()
  // CHECK-NEXT:   vm.global.store.i32 %0, @g0
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
  // CHECK: vm.global.ref public mutable @g0 : !vm.ref<?>
  vm.global.ref mutable @g0 initializer(@g0init) : !vm.ref<?>
  vm.func @g0init() -> !vm.ref<?> {
    %null = vm.const.ref.zero : !vm.ref<?>
    vm.return %null : !vm.ref<?>
  }

  // CHECK: vm.global.ref public mutable @g1 : !vm.ref<?>
  vm.global.ref mutable @g1 : !vm.ref<?>

  // CHECK: vm.global.ref public @g2 : !vm.ref<?>
  vm.global.ref @g2 : !vm.ref<?>

  // CHECK: vm.func private @__init() {
  // CHECK-NEXT:   %ref = vm.call @g0init()
  // CHECK-NEXT:   vm.global.store.ref %ref, @g0
  // CHECK-NEXT:   vm.return
  // CHECK-NEXT: }
}

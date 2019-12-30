// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-vm-global-initialization)' %s | IreeFileCheck %s

// CHECK: vm.module @initEmpty {
// CHECK: }
vm.module @initEmpty {
}

// -----

// CHECK-LABEL: @initI32
vm.module @initI32 {
  // CHECK: vm.global.i32 @g0 mutable : i32
  vm.global.i32 @g0 mutable init(@g0init) : i32
  vm.func @g0init() -> i32 {
    %c123 = vm.const.i32 123 : i32
    vm.return %c123 : i32
  }

  // CHECK: vm.global.i32 @g1 mutable : i32
  vm.global.i32 @g1 mutable 123 : i32

  // CHECK: vm.global.i32 @g2 mutable : i32
  vm.global.i32 @g2 123 : i32

  // CHECK: vm.func @__init() {
  // CHECK-NEXT:   %0 = vm.call @g0init()
  // CHECK-NEXT:   vm.global.store.i32 @g0, %0
  // CHECK-NEXT:   %c123 = vm.const.i32 123 : i32
  // CHECK-NEXT:   vm.global.store.i32 @g1, %c123
  // CHECK-NEXT:   %c123_0 = vm.const.i32 123 : i32
  // CHECK-NEXT:   vm.global.store.i32 @g2, %c123_0
  // CHECK-NEXT:   vm.return
  // CHECK-NEXT: }
}

// -----

// CHECK-LABEL: @initRef
vm.module @initRef {
  // CHECK: vm.global.ref @g0 mutable : !ireex.opaque_ref
  vm.global.ref @g0 mutable init(@g0init) : !ireex.opaque_ref
  vm.func @g0init() -> !ireex.opaque_ref {
    %null = vm.const.ref.zero : !ireex.opaque_ref
    vm.return %null : !ireex.opaque_ref
  }

  // CHECK: vm.global.ref @g1 mutable : !ireex.opaque_ref
  vm.global.ref @g1 mutable : !ireex.opaque_ref

  // CHECK: vm.global.ref @g2 : !ireex.opaque_ref
  vm.global.ref @g2 : !ireex.opaque_ref

  // CHECK: vm.func @__init() {
  // CHECK-NEXT:   %ref = vm.call @g0init()
  // CHECK-NEXT:   vm.global.store.ref @g0, %ref
  // CHECK-NEXT:   vm.return
  // CHECK-NEXT: }
}

// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc)' %s | FileCheck %s


vm.module @my_module {
  // CHECK-LABEL: @my_module_const_i32_zero
  vm.func @const_i32_zero() -> i32 {
    // CHECK: %[[ZERO:.+]] = "emitc.constant"() {value = 0 : i32} : () -> i32
    %zero = vm.const.i32.zero : i32
    vm.return %zero : i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_const_i32
  vm.func @const_i32() {
    // CHECK-NEXT: %0 = "emitc.constant"() {value = 0 : i32} : () -> i32
    %0 = vm.const.i32 0 : i32
    // CHECK-NEXT: %1 = "emitc.constant"() {value = 2 : i32} : () -> i32
    %1 = vm.const.i32 2 : i32
    // CHECK-NEXT: %2 = "emitc.constant"() {value = -2 : i32} : () -> i32
    %2 = vm.const.i32 -2 : i32
    vm.return
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_const_ref_zero
  vm.func @const_ref_zero() {
    // CHECK: %[[SIZE:.+]] = emitc.call "sizeof"() {args = [!emitc.opaque<"iree_vm_ref_t">]} : () -> i32
    // CHECK-NEXT: %[[VOIDPTR:.+]] = emitc.call "iree_alloca"(%[[SIZE]]) : (i32) -> !emitc.opaque<"void*">
    // CHECK-NEXT: %[[REFPTR:.+]] = emitc.call "EMITC_CAST"(%[[VOIDPTR]]) {args = [0 : index, !emitc.opaque<"iree_vm_ref_t*">]} : (!emitc.opaque<"void*">) -> !emitc.opaque<"iree_vm_ref_t*">
    // CHECK-NEXT: %[[SIZE_2:.+]] = emitc.call "sizeof"() {args = [#emitc.opaque<"iree_vm_ref_t">]} : () -> i32
    // CHECK-NEXT: emitc.call "memset"(%[[REFPTR]], %[[SIZE_2]]) {args = [0 : index, 0 : ui32, 1 : index]} : (!emitc.opaque<"iree_vm_ref_t*">, i32) -> ()
    // CHECK-NEXT: emitc.call "iree_vm_ref_release"(%[[REFPTR]]) : (!emitc.opaque<"iree_vm_ref_t*">) -> ()
    %null = vm.const.ref.zero : !vm.ref<?>
    vm.return
  }
}

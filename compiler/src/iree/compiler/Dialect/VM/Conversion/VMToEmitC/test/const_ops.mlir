// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s


vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_const_i32_zero
  vm.func @const_i32_zero() -> i32 {
    // CHECK: %[[ZERO:.+]] = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %zero = vm.const.i32.zero
    vm.return %zero : i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_const_i32
  vm.func @const_i32() {
    // CHECK-NEXT: %0 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %0 = vm.const.i32 0
    // CHECK-NEXT: %1 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1 = vm.const.i32 2
    // CHECK-NEXT: %2 = "emitc.constant"() <{value = -2 : i32}> : () -> i32
    %2 = vm.const.i32 -2
    vm.return
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_const_ref_zero
  vm.func @const_ref_zero() {
    // CHECK: %[[REF:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %[[REFPTR:.+]] = emitc.apply "&"(%[[REF]]) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
    // CHECK-NEXT: %[[SIZE:.+]] = emitc.call_opaque "sizeof"() {args = [!emitc.opaque<"iree_vm_ref_t">]} : () -> !emitc.opaque<"iree_host_size_t">
    // CHECK-NEXT: emitc.call_opaque "memset"(%[[REFPTR]], %[[SIZE]]) {args = [0 : index, 0 : ui32, 1 : index]} : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, !emitc.opaque<"iree_host_size_t">) -> ()
    // CHECK-NEXT: emitc.call_opaque "iree_vm_ref_release"(%[[REFPTR]]) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> ()
    %null = vm.const.ref.zero : !vm.ref<?>
    vm.return
  }
}

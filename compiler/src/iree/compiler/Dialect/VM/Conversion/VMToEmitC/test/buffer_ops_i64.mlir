// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

vm.module @my_module {
  // CHECK-LABEL: @my_module_buffer_fill_i64
  vm.func @buffer_fill_i64(%buf : !vm.buffer) {
    // CHECK: %[[C0:.+]] = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    // CHECK: %[[C16:.+]] = "emitc.constant"() <{value = 16 : i64}> : () -> i64
    // CHECK: %[[C102:.+]] = "emitc.constant"() <{value = 102 : i64}> : () -> i64

    // CHECK: %[[BUFFER_REF:.+]] = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %[[BUFFER_PTR:.+]] = emitc.call_opaque "iree_vm_buffer_deref"(%[[BUFFER_REF]]) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>

    // CHECK: %[[STATUS:.+]] = emitc.call_opaque "vm_buffer_fill_i64"(%[[BUFFER_PTR]], %[[C0]], %[[C16]], %[[C102]]) : (!emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>, i64, i64, i64) -> !emitc.opaque<"iree_status_t">
    %c0 = vm.const.i64 0
    %c16 = vm.const.i64 16
    %c102 = vm.const.i64 102
    vm.buffer.fill.i64 %buf, %c0, %c16, %c102 : i64 -> !vm.buffer
    vm.return
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_buffer_load_i64
  vm.func @buffer_load_i64(%buf : !vm.buffer) {
    // CHECK: %[[C0:.+]] = "emitc.constant"() <{value = 0 : i64}> : () -> i64

    // CHECK: %[[BUFFER_REF:.+]] = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %[[BUFFER_PTR:.+]] = emitc.call_opaque "iree_vm_buffer_deref"(%[[BUFFER_REF]]) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>

    // CHECK: %[[RESULT:.+]] = "emitc.variable"() <{value = 0 : i64}> : () -> i64
    // CHECK-NEXT: %[[RESULT_PTR:.+]] = emitc.apply "&"(%[[RESULT]]) : (i64) -> !emitc.ptr<i64>
    // CHECK-NEXT: %[[STATUS:.+]] = emitc.call_opaque "vm_buffer_load_i64"(%[[BUFFER_PTR]], %[[C0]], %[[RESULT_PTR]]) : (!emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>, i64, !emitc.ptr<i64>) -> !emitc.opaque<"iree_status_t">

    %c0 = vm.const.i64 0
    %v0 = vm.buffer.load.i64 %buf[%c0] : !vm.buffer -> i64
    vm.return
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_buffer_store_i64
  vm.func @buffer_store_i64(%buf : !vm.buffer) {
    // CHECK: %[[C0:.+]] = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    // CHECK: %[[C102:.+]] = "emitc.constant"() <{value = 102 : i64}> : () -> i64

    // CHECK: %[[BUFFER_REF:.+]] = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %[[BUFFER_PTR:.+]] = emitc.call_opaque "iree_vm_buffer_deref"(%[[BUFFER_REF]]) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>

    // CHECK: %[[STATUS:.+]] = emitc.call_opaque "vm_buffer_store_i64"(%[[BUFFER_PTR]], %[[C0]], %[[C102]]) : (!emitc.ptr<!emitc.opaque<"iree_vm_buffer_t">>, i64, i64) -> !emitc.opaque<"iree_status_t">
    %c0 = vm.const.i64 0
    %c102 = vm.const.i64 102
    vm.buffer.store.i64 %c102, %buf[%c0] : i64 -> !vm.buffer
    vm.return
  }
}

// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

vm.module @my_module {
  vm.global.i32 private @c42 = 42 : i32

  // CHECK-LABEL: emitc.func private @my_module_global_load_i32
  vm.func @global_load_i32() -> i32 {
    // CHECK-NEXT: %[[STATE_LVAL:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>
    // CHECK-NEXT: emitc.assign %arg2 : !emitc.ptr<!emitc.opaque<"struct my_module_state_t">> to %[[STATE_LVAL]] : <!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>
    // CHECK-NEXT: %[[RWDATA_LVAL:.+]] = "emitc.member_of_ptr"(%[[STATE_LVAL]]) <{member = "rwdata"}> : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>) -> !emitc.lvalue<!emitc.ptr<ui8>>
    // CHECK-NEXT: %[[RWDATA:.+]] = emitc.load %[[RWDATA_LVAL]] : <!emitc.ptr<ui8>>
    // CHECK-NEXT: %[[RES:.+]] = emitc.call_opaque "vm_global_load_i32"(%[[RWDATA]]) {args = [0 : index, 0 : ui32]} : (!emitc.ptr<ui8>) -> i32
    %0 = vm.global.load.i32 @c42 : i32
    vm.return %0 : i32
  }
}

// -----

vm.module @my_module {
  vm.global.i32 private mutable @c107_mut = 107 : i32

  // CHECK-LABEL: emitc.func private @my_module_global_store_i32
  vm.func @global_store_i32(%arg0 : i32) {
    // CHECK-NEXT: %[[STATE_LVAL:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>
    // CHECK-NEXT: emitc.assign %arg2 : !emitc.ptr<!emitc.opaque<"struct my_module_state_t">> to %[[STATE_LVAL]] : <!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>
    // CHECK-NEXT: %[[RWDATA_LVAL:.+]] = "emitc.member_of_ptr"(%[[STATE_LVAL]]) <{member = "rwdata"}> : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>) -> !emitc.lvalue<!emitc.ptr<ui8>>
    // CHECK-NEXT: %[[RWDATA:.+]] = emitc.load %[[RWDATA_LVAL]] : <!emitc.ptr<ui8>>
    // CHECK-NEXT: emitc.call_opaque "vm_global_store_i32"(%[[RWDATA]], %arg3) {args = [0 : index, 0 : ui32, 1 : index]} : (!emitc.ptr<ui8>, i32) -> ()
    vm.global.store.i32 %arg0, @c107_mut : i32
    vm.return
  }
}

// -----

vm.module @my_module {
  vm.global.ref private @g0 : !vm.buffer

  // CHECK-LABEL: emitc.func private @my_module_global_load_ref
  vm.func @global_load_ref() -> !vm.buffer {
    // CHECK: %[[STATE_LVAL:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>
    // CHECK-NEXT: emitc.assign %arg2 : !emitc.ptr<!emitc.opaque<"struct my_module_state_t">> to %[[STATE_LVAL]] : <!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>
    // CHECK-NEXT: %[[REFS_LVAL:.+]] = "emitc.member_of_ptr"(%[[STATE_LVAL]]) <{member = "refs"}> : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>) -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>>
    // CHECK-NEXT: %[[REFS:.+]] = emitc.load %[[REFS_LVAL]] : <!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>>
    // CHECK: %[[REF_INDEX:.+]] = emitc.literal "0" : !emitc.opaque<"iree_host_size_t">
    // CHECK-NEXT: %[[REF:.+]] = emitc.subscript %[[REFS]][%[[REF_INDEX]]] : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, !emitc.opaque<"iree_host_size_t">) -> !emitc.lvalue<!emitc.opaque<"iree_vm_ref_t">>
    // CHECK-NEXT: %[[REF_0:.+]] = emitc.apply "&"(%[[REF]]) : (!emitc.lvalue<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
    // CHECK: %[[C:.+]] = emitc.call_opaque "iree_vm_type_def_as_ref"(%{{.+}}) : (!emitc.opaque<"iree_vm_type_def_t">) -> !emitc.opaque<"iree_vm_ref_type_t">
    // CHECK: %{{.+}} = emitc.call_opaque "iree_vm_ref_retain_or_move_checked"(%[[REF_0]], %[[C]], %arg3) {args = [false, 0 : index, 1 : index, 2 : index]} : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, !emitc.opaque<"iree_vm_ref_type_t">, !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_status_t">
    %0 = vm.global.load.ref @g0 : !vm.buffer
    vm.return %0 : !vm.buffer
  }
}

// -----

vm.module @my_module {
  vm.global.ref private mutable @g0_mut : !vm.buffer

  // CHECK-LABEL: emitc.func private @my_module_global_store_ref
  vm.func @global_store_ref(%arg0 : !vm.buffer) {
    // CHECK: %[[STATE_LVAL:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>
    // CHECK-NEXT: emitc.assign %arg2 : !emitc.ptr<!emitc.opaque<"struct my_module_state_t">> to %[[STATE_LVAL]] : <!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>
    // CHECK-NEXT: %[[REFS_LVAL:.+]] = "emitc.member_of_ptr"(%[[STATE_LVAL]]) <{member = "refs"}> : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>) -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>>
    // CHECK-NEXT: %[[REFS:.+]] = emitc.load %[[REFS_LVAL]] : <!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>>
    // CHECK: %[[REF_INDEX:.+]] = emitc.literal "0" : !emitc.opaque<"iree_host_size_t">
    // CHECK-NEXT: %[[REF:.+]] = emitc.subscript %[[REFS]][%[[REF_INDEX]]] : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, !emitc.opaque<"iree_host_size_t">) -> !emitc.lvalue<!emitc.opaque<"iree_vm_ref_t">>
    // CHECK-NEXT: %[[REF_0:.+]] = emitc.apply "&"(%[[REF]]) : (!emitc.lvalue<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
    // CHECK: %[[C:.+]] = emitc.call_opaque "iree_vm_type_def_as_ref"(%{{.+}}) : (!emitc.opaque<"iree_vm_type_def_t">) -> !emitc.opaque<"iree_vm_ref_type_t">
    // CHECK: %{{.+}} = emitc.call_opaque "iree_vm_ref_retain_or_move_checked"(%arg3, %[[C]], %[[REF_0]]) {args = [false, 0 : index, 1 : index, 2 : index]} : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, !emitc.opaque<"iree_vm_ref_type_t">, !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_status_t">
    vm.global.store.ref %arg0, @g0_mut : !vm.buffer
    vm.return
  }
}

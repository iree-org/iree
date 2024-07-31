// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

vm.module @my_module {
  vm.global.i64 private @c42 = 42 : i64

  // CHECK-LABEL: emitc.func private @my_module_global_load_i64
  vm.func @global_load_i64() -> i64 {
    // CHECK-NEXT: %[[STATE_LVAL:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>
    // CHECK-NEXT: emitc.assign %arg2 : !emitc.ptr<!emitc.opaque<"struct my_module_state_t">> to %[[STATE_LVAL]] : <!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>
    // CHECK-NEXT: %[[RWDATA_LVAL:.+]] = "emitc.member_of_ptr"(%[[STATE_LVAL]]) <{member = "rwdata"}> : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>) -> !emitc.lvalue<!emitc.ptr<ui8>>
    // CHECK-NEXT: %[[RWDATA:.+]] = emitc.load %[[RWDATA_LVAL]] : <!emitc.ptr<ui8>>
    // CHECK-NEXT: %[[RES:.+]] = emitc.call_opaque "vm_global_load_i64"(%[[RWDATA]]) {args = [0 : index, 0 : ui32]} : (!emitc.ptr<ui8>) -> i64
    %0 = vm.global.load.i64 @c42 : i64
    vm.return %0 : i64
  }
}

// -----

vm.module @my_module {
  vm.global.i64 private mutable @c107_mut = 107 : i64

  // CHECK-LABEL: emitc.func private @my_module_global_store_i64
  vm.func @global_store_i64(%arg0 : i64) {
    // CHECK-NEXT: %[[STATE_LVAL:.+]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>
    // CHECK-NEXT: emitc.assign %arg2 : !emitc.ptr<!emitc.opaque<"struct my_module_state_t">> to %[[STATE_LVAL]] : <!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>
    // CHECK-NEXT: %[[RWDATA_LVAL:.+]] = "emitc.member_of_ptr"(%[[STATE_LVAL]]) <{member = "rwdata"}> : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>>) -> !emitc.lvalue<!emitc.ptr<ui8>>
    // CHECK-NEXT: %[[RWDATA:.+]] = emitc.load %[[RWDATA_LVAL]] : <!emitc.ptr<ui8>>
    // CHECK-NEXT: emitc.call_opaque "vm_global_store_i64"(%[[RWDATA]], %arg3) {args = [0 : index, 0 : ui32, 1 : index]} : (!emitc.ptr<ui8>, i64) -> ()
    vm.global.store.i64 %arg0, @c107_mut : i64
    vm.return
  }
}

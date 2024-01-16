// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

vm.module @my_module {
  vm.global.i64 private @c42 = 42 : i64

  // CHECK-LABEL: @my_module_global_load_i64
  vm.func @global_load_i64() -> i64 {
    // CHECK-NEXT: %0 = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"rwdata">]} : (!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>) -> !emitc.ptr<ui8>
    // CHECK-NEXT: %1 = emitc.call_opaque "vm_global_load_i64"(%0) {args = [0 : index, 0 : ui32]} : (!emitc.ptr<ui8>) -> i64
    %0 = vm.global.load.i64 @c42 : i64
    vm.return %0 : i64
  }
}

// -----

vm.module @my_module {
  vm.global.i64 private mutable @c107_mut = 107 : i64

  // CHECK-LABEL: @my_module_global_store_i64
  vm.func @global_store_i64(%arg0 : i64) {
    // CHECK-NEXT: %0 = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"rwdata">]} : (!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>) -> !emitc.ptr<ui8>
    // CHECK-NEXT: emitc.call_opaque "vm_global_store_i64"(%0, %arg3) {args = [0 : index, 0 : ui32, 1 : index]} : (!emitc.ptr<ui8>, i64) -> ()
    vm.global.store.i64 %arg0, @c107_mut : i64
    vm.return
  }
}

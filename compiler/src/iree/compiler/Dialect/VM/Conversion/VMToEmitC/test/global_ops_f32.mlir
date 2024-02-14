// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

vm.module @my_module {
  vm.global.f32 private @c42 = 42.5 : f32

  // CHECK-LABEL: emitc.func private @my_module_global_load_f32
  vm.func @global_load_f32() -> f32 {
    // CHECK-NEXT: %0 = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"rwdata">]} : (!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>) -> !emitc.ptr<ui8>
    // CHECK-NEXT: %1 = emitc.call_opaque "vm_global_load_f32"(%0) {args = [0 : index, 0 : ui32]} : (!emitc.ptr<ui8>) -> f32
    %0 = vm.global.load.f32 @c42 : f32
    vm.return %0 : f32
  }
}

// -----

vm.module @my_module {
  vm.global.f32 private mutable @c107_mut = 107.5 : f32

  // CHECK-LABEL: emitc.func private @my_module_global_store_f32
  vm.func @global_store_f32(%arg0 : f32) {
    // CHECK-NEXT: %0 = emitc.call_opaque "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"rwdata">]} : (!emitc.ptr<!emitc.opaque<"struct my_module_state_t">>) -> !emitc.ptr<ui8>
    // CHECK-NEXT: emitc.call_opaque "vm_global_store_f32"(%0, %arg3) {args = [0 : index, 0 : ui32, 1 : index]} : (!emitc.ptr<ui8>, f32) -> ()
    vm.global.store.f32 %arg0, @c107_mut : f32
    vm.return
  }
}

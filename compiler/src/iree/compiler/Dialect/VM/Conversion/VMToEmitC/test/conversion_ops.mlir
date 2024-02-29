// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

// CHECK-LABEL: emitc.func private @my_module_trunc
vm.module @my_module {
  vm.func @trunc(%arg0 : i32) -> i32 {
    // CHECK-NEXT: %0 = emitc.call_opaque "vm_trunc_i32i8"(%arg3) : (i32) -> i32
    %0 = vm.trunc.i32.i8 %arg0 : i32 -> i32
    // CHECK-NEXT: %1 = emitc.call_opaque "vm_trunc_i32i16"(%0) : (i32) -> i32
    %1 = vm.trunc.i32.i16 %0 : i32 -> i32
    vm.return %1 : i32
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_ext
vm.module @my_module {
  vm.func @ext(%arg0 : i32) -> i32 {
    // CHECK-NEXT: %0 = emitc.call_opaque "vm_ext_i8i32s"(%arg3) : (i32) -> i32
    %0 = vm.ext.i8.i32.s %arg0 : i32 -> i32
    // CHECK-NEXT: %1 = emitc.call_opaque "vm_ext_i8i32u"(%0) : (i32) -> i32
    %1 = vm.ext.i8.i32.u %0 : i32 -> i32
    // CHECK-NEXT: %2 = emitc.call_opaque "vm_ext_i16i32s"(%1) : (i32) -> i32
    %2 = vm.ext.i16.i32.s %1 : i32 -> i32
    // CHECK-NEXT: %3 = emitc.call_opaque "vm_ext_i16i32u"(%2) : (i32) -> i32
    %3 = vm.ext.i16.i32.u %2 : i32 -> i32
    vm.return %3 : i32
  }
}

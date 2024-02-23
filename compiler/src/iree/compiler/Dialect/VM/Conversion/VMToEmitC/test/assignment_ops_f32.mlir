// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

// CHECK-LABEL: emitc.func private @my_module_select_f32
vm.module @my_module {
  vm.func @select_f32(%arg0 : i32, %arg1 : f32, %arg2 : f32) -> f32 {
    // CHECK: %0 = emitc.call_opaque "vm_select_f32"(%arg3, %arg4, %arg5) : (i32, f32, f32) -> f32
    %0 = vm.select.f32 %arg0, %arg1, %arg2 : f32
    vm.return %0 : f32
  }
}

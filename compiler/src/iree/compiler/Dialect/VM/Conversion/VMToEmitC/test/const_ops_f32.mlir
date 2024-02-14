// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_const_f32_zero
  vm.func @const_f32_zero() -> f32 {
    // CHECK: %[[ZERO:.+]] = "emitc.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %zero = vm.const.f32.zero
    vm.return %zero : f32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_const_f32
  vm.func @const_f32() {
    // CHECK-NEXT: %0 = "emitc.constant"() <{value = 5.000000e-01 : f32}> : () -> f32
    %0 = vm.const.f32 0.5
    // CHECK-NEXT: %1 = "emitc.constant"() <{value = 2.500000e+00 : f32}> : () -> f32
    %1 = vm.const.f32 2.5
    // CHECK-NEXT: %2 = "emitc.constant"() <{value = -2.500000e+00 : f32}> : () -> f32
    %2 = vm.const.f32 -2.5
    vm.return
  }
}

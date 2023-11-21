// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

// CHECK-LABEL: @my_module_shl_i32
vm.module @my_module {
  vm.func @shl_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = emitc.call_opaque "vm_shl_i32"(%arg3, %arg4) : (i32, i32) -> i32
    %0 = vm.shl.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @my_module_shr_i32_s
vm.module @my_module {
  vm.func @shr_i32_s(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = emitc.call_opaque "vm_shr_i32s"(%arg3, %arg4) : (i32, i32) -> i32
    %0 = vm.shr.i32.s %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @my_module_shr_i32_u
vm.module @my_module {
  vm.func @shr_i32_u(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = emitc.call_opaque "vm_shr_i32u"(%arg3, %arg4) : (i32, i32) -> i32
    %0 = vm.shr.i32.u %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

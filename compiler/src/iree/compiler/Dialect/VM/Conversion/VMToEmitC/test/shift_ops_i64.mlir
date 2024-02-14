// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

// CHECK-LABEL: emitc.func private @my_module_shl_i64
vm.module @my_module {
  vm.func @shl_i64(%arg0 : i64, %arg1 : i32) -> i64 {
    // CHECK: %0 = emitc.call_opaque "vm_shl_i64"(%arg3, %arg4) : (i64, i32) -> i64
    %0 = vm.shl.i64 %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_shr_i64_s
vm.module @my_module {
  vm.func @shr_i64_s(%arg0 : i64, %arg1 : i32) -> i64 {
    // CHECK: %0 = emitc.call_opaque "vm_shr_i64s"(%arg3, %arg4) : (i64, i32) -> i64
    %0 = vm.shr.i64.s %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_shr_i64_u
vm.module @my_module {
  vm.func @shr_i64_u(%arg0 : i64, %arg1 : i32) -> i64 {
    // CHECK: %0 = emitc.call_opaque "vm_shr_i64u"(%arg3, %arg4) : (i64, i32) -> i64
    %0 = vm.shr.i64.u %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

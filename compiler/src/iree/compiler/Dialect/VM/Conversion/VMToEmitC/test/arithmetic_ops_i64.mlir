// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

// CHECK-LABEL: emitc.func private @my_module_add_i64
vm.module @my_module {
  vm.func @add_i64(%arg0: i64, %arg1: i64) {
    // CHECK-NEXT: %0 = emitc.call_opaque "vm_add_i64"(%arg3, %arg4) : (i64, i64) -> i64
    %0 = vm.add.i64 %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_sub_i64
vm.module @my_module {
  vm.func @sub_i64(%arg0: i64, %arg1: i64) {
    // CHECK: %0 = emitc.call_opaque "vm_sub_i64"(%arg3, %arg4) : (i64, i64) -> i64
    %0 = vm.sub.i64 %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_mul_i64
vm.module @my_module {
  vm.func @mul_i64(%arg0: i64, %arg1: i64) {
    // CHECK: %0 = emitc.call_opaque "vm_mul_i64"(%arg3, %arg4) : (i64, i64) -> i64
    %0 = vm.mul.i64 %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_div_i64_s
vm.module @my_module {
  vm.func @div_i64_s(%arg0: i64, %arg1: i64) {
    // CHECK: %0 = emitc.call_opaque "vm_div_i64s"(%arg3, %arg4) : (i64, i64) -> i64
    %0 = vm.div.i64.s %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_div_i64_u
vm.module @my_module {
  vm.func @div_i64_u(%arg0: i64, %arg1: i64) {
    // CHECK: %0 = emitc.call_opaque "vm_div_i64u"(%arg3, %arg4) : (i64, i64) -> i64
    %0 = vm.div.i64.u %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_rem_i64_s
vm.module @my_module {
  vm.func @rem_i64_s(%arg0: i64, %arg1: i64) {
    // CHECK: %0 = emitc.call_opaque "vm_rem_i64s"(%arg3, %arg4) : (i64, i64) -> i64
    %0 = vm.rem.i64.s %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_rem_i64_u
vm.module @my_module {
  vm.func @rem_i64_u(%arg0: i64, %arg1: i64) {
    // CHECK: %0 = emitc.call_opaque "vm_rem_i64u"(%arg3, %arg4) : (i64, i64) -> i64
    %0 = vm.rem.i64.u %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_fma_i64
vm.module @my_module {
  vm.func @fma_i64(%arg0: i64, %arg1: i64, %arg2: i64) {
    // CHECK: %0 = emitc.call_opaque "vm_fma_i64"(%arg3, %arg4, %arg5) : (i64, i64, i64) -> i64
    %0 = vm.fma.i64 %arg0, %arg1, %arg2 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_abs_i64
vm.module @my_module {
  vm.func @abs_i64(%arg0 : i64) -> i64 {
    // CHECK: %0 = emitc.call_opaque "vm_abs_i64"(%arg3) : (i64) -> i64
    %0 = vm.abs.i64 %arg0 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_min_i64_s
vm.module @my_module {
  vm.func @min_i64_s(%arg0: i64, %arg1: i64) {
    // CHECK: %0 = emitc.call_opaque "vm_min_i64s"(%arg3, %arg4) : (i64, i64) -> i64
    %0 = vm.min.i64.s %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_not_i64
vm.module @my_module {
  vm.func @not_i64(%arg0 : i64) -> i64 {
    // CHECK: %0 = emitc.call_opaque "vm_not_i64"(%arg3) : (i64) -> i64
    %0 = vm.not.i64 %arg0 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_and_i64
vm.module @my_module {
  vm.func @and_i64(%arg0 : i64, %arg1 : i64) -> i64 {
    // CHECK: %0 = emitc.call_opaque "vm_and_i64"(%arg3, %arg4) : (i64, i64) -> i64
    %0 = vm.and.i64 %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_or_i64
vm.module @my_module {
  vm.func @or_i64(%arg0 : i64, %arg1 : i64) -> i64 {
    // CHECK: %0 = emitc.call_opaque "vm_or_i64"(%arg3, %arg4) : (i64, i64) -> i64
    %0 = vm.or.i64 %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_xor_i64
vm.module @my_module {
  vm.func @xor_i64(%arg0 : i64, %arg1 : i64) -> i64 {
    // CHECK: %0 = emitc.call_opaque "vm_xor_i64"(%arg3, %arg4) : (i64, i64) -> i64
    %0 = vm.xor.i64 %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

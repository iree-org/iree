// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

// CHECK-LABEL: emitc.func private @my_module_add_i32
vm.module @my_module {
  vm.func @add_i32(%arg0: i32, %arg1: i32) {
    // CHECK-NEXT: %0 = emitc.call_opaque "vm_add_i32"(%arg3, %arg4) : (i32, i32) -> i32
    %0 = vm.add.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_sub_i32
vm.module @my_module {
  vm.func @sub_i32(%arg0: i32, %arg1: i32) {
    // CHECK: %0 = emitc.call_opaque "vm_sub_i32"(%arg3, %arg4) : (i32, i32) -> i32
    %0 = vm.sub.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_mul_i32
vm.module @my_module {
  vm.func @mul_i32(%arg0: i32, %arg1: i32) {
    // CHECK: %0 = emitc.call_opaque "vm_mul_i32"(%arg3, %arg4) : (i32, i32) -> i32
    %0 = vm.mul.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_div_i32_s
vm.module @my_module {
  vm.func @div_i32_s(%arg0: i32, %arg1: i32) {
    // CHECK: %0 = emitc.call_opaque "vm_div_i32s"(%arg3, %arg4) : (i32, i32) -> i32
    %0 = vm.div.i32.s %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_div_i32_u
vm.module @my_module {
  vm.func @div_i32_u(%arg0: i32, %arg1: i32) {
    // CHECK: %0 = emitc.call_opaque "vm_div_i32u"(%arg3, %arg4) : (i32, i32) -> i32
    %0 = vm.div.i32.u %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_rem_i32_s
vm.module @my_module {
  vm.func @rem_i32_s(%arg0: i32, %arg1: i32) {
    // CHECK: %0 = emitc.call_opaque "vm_rem_i32s"(%arg3, %arg4) : (i32, i32) -> i32
    %0 = vm.rem.i32.s %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_rem_i32_u
vm.module @my_module {
  vm.func @rem_i32_u(%arg0: i32, %arg1: i32) {
    // CHECK: %0 = emitc.call_opaque "vm_rem_i32u"(%arg3, %arg4) : (i32, i32) -> i32
    %0 = vm.rem.i32.u %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_fma_i32
vm.module @my_module {
  vm.func @fma_i32(%arg0: i32, %arg1: i32, %arg2: i32) {
    // CHECK: %0 = emitc.call_opaque "vm_fma_i32"(%arg3, %arg4, %arg5) : (i32, i32, i32) -> i32
    %0 = vm.fma.i32 %arg0, %arg1, %arg2 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_abs_i32
vm.module @my_module {
  vm.func @abs_i32(%arg0 : i32) -> i32 {
    // CHECK: %0 = emitc.call_opaque "vm_abs_i32"(%arg3) : (i32) -> i32
    %0 = vm.abs.i32 %arg0 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_min_i32_s
vm.module @my_module {
  vm.func @min_i32_s(%arg0: i32, %arg1: i32) {
    // CHECK: %0 = emitc.call_opaque "vm_min_i32s"(%arg3, %arg4) : (i32, i32) -> i32
    %0 = vm.min.i32.s %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_not_i32
vm.module @my_module {
  vm.func @not_i32(%arg0 : i32) -> i32 {
    // CHECK: %0 = emitc.call_opaque "vm_not_i32"(%arg3) : (i32) -> i32
    %0 = vm.not.i32 %arg0 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_and_i32
vm.module @my_module {
  vm.func @and_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = emitc.call_opaque "vm_and_i32"(%arg3, %arg4) : (i32, i32) -> i32
    %0 = vm.and.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_or_i32
vm.module @my_module {
  vm.func @or_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = emitc.call_opaque "vm_or_i32"(%arg3, %arg4) : (i32, i32) -> i32
    %0 = vm.or.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: emitc.func private @my_module_xor_i32
vm.module @my_module {
  vm.func @xor_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = emitc.call_opaque "vm_xor_i32"(%arg3, %arg4) : (i32, i32) -> i32
    %0 = vm.xor.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

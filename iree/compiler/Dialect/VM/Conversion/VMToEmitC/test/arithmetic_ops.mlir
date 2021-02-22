// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-convert-vm-to-emitc)' %s | IreeFileCheck %s

// CHECK-LABEL: @add_i32
vm.module @my_module {
  vm.func @add_i32(%arg0: i32, %arg1: i32) {
    // CHECK-NEXT: %0 = emitc.call "vm_add_i32"(%arg0, %arg1) : (i32, i32) -> i32
    %0 = vm.add.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @sub_i32
vm.module @my_module {
  vm.func @sub_i32(%arg0: i32, %arg1: i32) {
    // CHECK: %0 = emitc.call "vm_sub_i32"(%arg0, %arg1) : (i32, i32) -> i32
    %0 = vm.sub.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @mul_i32
vm.module @my_module {
  vm.func @mul_i32(%arg0: i32, %arg1: i32) {
    // CHECK: %0 = emitc.call "vm_mul_i32"(%arg0, %arg1) : (i32, i32) -> i32
    %0 = vm.mul.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @div_i32_s
vm.module @my_module {
  vm.func @div_i32_s(%arg0: i32, %arg1: i32) {
    // CHECK: %0 = emitc.call "vm_div_i32s"(%arg0, %arg1) : (i32, i32) -> i32
    %0 = vm.div.i32.s %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @div_i32_u
vm.module @my_module {
  vm.func @div_i32_u(%arg0: i32, %arg1: i32) {
    // CHECK: %0 = emitc.call "vm_div_i32u"(%arg0, %arg1) : (i32, i32) -> i32
    %0 = vm.div.i32.u %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @rem_i32_s
vm.module @my_module {
  vm.func @rem_i32_s(%arg0: i32, %arg1: i32) {
    // CHECK: %0 = emitc.call "vm_rem_i32s"(%arg0, %arg1) : (i32, i32) -> i32
    %0 = vm.rem.i32.s %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @rem_i32_u
vm.module @my_module {
  vm.func @rem_i32_u(%arg0: i32, %arg1: i32) {
    // CHECK: %0 = emitc.call "vm_rem_i32u"(%arg0, %arg1) : (i32, i32) -> i32
    %0 = vm.rem.i32.u %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @not_i32
vm.module @my_module {
  vm.func @not_i32(%arg0 : i32) -> i32 {
    // CHECK: %0 = emitc.call "vm_not_i32"(%arg0) : (i32) -> i32
    %0 = vm.not.i32 %arg0 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @and_i32
vm.module @my_module {
  vm.func @and_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = emitc.call "vm_and_i32"(%arg0, %arg1) : (i32, i32) -> i32
    %0 = vm.and.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @or_i32
vm.module @my_module {
  vm.func @or_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = emitc.call "vm_or_i32"(%arg0, %arg1) : (i32, i32) -> i32
    %0 = vm.or.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @xor_i32
vm.module @my_module {
  vm.func @xor_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = emitc.call "vm_xor_i32"(%arg0, %arg1) : (i32, i32) -> i32
    %0 = vm.xor.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

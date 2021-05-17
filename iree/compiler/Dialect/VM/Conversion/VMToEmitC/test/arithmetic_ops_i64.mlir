// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-convert-vm-to-emitc)' %s | IreeFileCheck %s

// CHECK-LABEL: @add_i64
vm.module @my_module {
  vm.func @add_i64(%arg0: i64, %arg1: i64) {
    // CHECK-NEXT: %0 = emitc.call "vm_add_i64"(%arg0, %arg1) : (i64, i64) -> i64
    %0 = vm.add.i64 %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: @sub_i64
vm.module @my_module {
  vm.func @sub_i64(%arg0: i64, %arg1: i64) {
    // CHECK: %0 = emitc.call "vm_sub_i64"(%arg0, %arg1) : (i64, i64) -> i64
    %0 = vm.sub.i64 %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: @mul_i64
vm.module @my_module {
  vm.func @mul_i64(%arg0: i64, %arg1: i64) {
    // CHECK: %0 = emitc.call "vm_mul_i64"(%arg0, %arg1) : (i64, i64) -> i64
    %0 = vm.mul.i64 %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: @div_i64_s
vm.module @my_module {
  vm.func @div_i64_s(%arg0: i64, %arg1: i64) {
    // CHECK: %0 = emitc.call "vm_div_i64s"(%arg0, %arg1) : (i64, i64) -> i64
    %0 = vm.div.i64.s %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: @div_i64_u
vm.module @my_module {
  vm.func @div_i64_u(%arg0: i64, %arg1: i64) {
    // CHECK: %0 = emitc.call "vm_div_i64u"(%arg0, %arg1) : (i64, i64) -> i64
    %0 = vm.div.i64.u %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: @rem_i64_s
vm.module @my_module {
  vm.func @rem_i64_s(%arg0: i64, %arg1: i64) {
    // CHECK: %0 = emitc.call "vm_rem_i64s"(%arg0, %arg1) : (i64, i64) -> i64
    %0 = vm.rem.i64.s %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: @rem_i64_u
vm.module @my_module {
  vm.func @rem_i64_u(%arg0: i64, %arg1: i64) {
    // CHECK: %0 = emitc.call "vm_rem_i64u"(%arg0, %arg1) : (i64, i64) -> i64
    %0 = vm.rem.i64.u %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: @fma_i64
vm.module @my_module {
  vm.func @fma_i64(%arg0: i64, %arg1: i64, %arg2: i64) {
    // CHECK: %0 = emitc.call "vm_fma_i64"(%arg0, %arg1, %arg2) : (i64, i64, i64) -> i64
    %0 = vm.fma.i64 %arg0, %arg1, %arg2 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: @not_i64
vm.module @my_module {
  vm.func @not_i64(%arg0 : i64) -> i64 {
    // CHECK: %0 = emitc.call "vm_not_i64"(%arg0) : (i64) -> i64
    %0 = vm.not.i64 %arg0 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: @and_i64
vm.module @my_module {
  vm.func @and_i64(%arg0 : i64, %arg1 : i64) -> i64 {
    // CHECK: %0 = emitc.call "vm_and_i64"(%arg0, %arg1) : (i64, i64) -> i64
    %0 = vm.and.i64 %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: @or_i64
vm.module @my_module {
  vm.func @or_i64(%arg0 : i64, %arg1 : i64) -> i64 {
    // CHECK: %0 = emitc.call "vm_or_i64"(%arg0, %arg1) : (i64, i64) -> i64
    %0 = vm.or.i64 %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: @xor_i64
vm.module @my_module {
  vm.func @xor_i64(%arg0 : i64, %arg1 : i64) -> i64 {
    // CHECK: %0 = emitc.call "vm_xor_i64"(%arg0, %arg1) : (i64, i64) -> i64
    %0 = vm.xor.i64 %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

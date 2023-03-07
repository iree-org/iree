// RUN: iree-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: @add_i32
vm.module @my_module {
  vm.func @add_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.add.i32 %arg0, %arg1 : i32
    %0 = vm.add.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @sub_i32
vm.module @my_module {
  vm.func @sub_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.sub.i32 %arg0, %arg1 : i32
    %0 = vm.sub.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @mul_i32
vm.module @my_module {
  vm.func @mul_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.mul.i32 %arg0, %arg1 : i32
    %0 = vm.mul.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @div_i32_s
vm.module @my_module {
  vm.func @div_i32_s(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.div.i32.s %arg0, %arg1 : i32
    %0 = vm.div.i32.s %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @div_i32_u
vm.module @my_module {
  vm.func @div_i32_u(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.div.i32.u %arg0, %arg1 : i32
    %0 = vm.div.i32.u %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @rem_i32_s
vm.module @my_module {
  vm.func @rem_i32_s(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.rem.i32.s %arg0, %arg1 : i32
    %0 = vm.rem.i32.s %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @rem_i32_u
vm.module @my_module {
  vm.func @rem_i32_u(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.rem.i32.u %arg0, %arg1 : i32
    %0 = vm.rem.i32.u %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @fma_i32
vm.module @my_module {
  vm.func @fma_i32(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // CHECK: %0 = vm.fma.i32 %arg0, %arg1, %arg2 : i32
    %0 = vm.fma.i32 %arg0, %arg1, %arg2 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @abs_i32
vm.module @my_module {
  vm.func @abs_i32(%arg0 : i32) -> i32 {
    // CHECK: %0 = vm.abs.i32 %arg0 : i32
    %0 = vm.abs.i32 %arg0 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @not_i32
vm.module @my_module {
  vm.func @not_i32(%arg0 : i32) -> i32 {
    // CHECK: %0 = vm.not.i32 %arg0 : i32
    %0 = vm.not.i32 %arg0 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @and_i32
vm.module @my_module {
  vm.func @and_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.and.i32 %arg0, %arg1 : i32
    %0 = vm.and.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @or_i32
vm.module @my_module {
  vm.func @or_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.or.i32 %arg0, %arg1 : i32
    %0 = vm.or.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @xor_i32
vm.module @my_module {
  vm.func @xor_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.xor.i32 %arg0, %arg1 : i32
    %0 = vm.xor.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @ctlz_i32
vm.module @my_module {
  vm.func @ctlz_i32(%arg0 : i32) -> i32 {
    // CHECK: %0 = vm.ctlz.i32 %arg0 : i32
    %0 = vm.ctlz.i32 %arg0 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @min_i32_s
vm.module @my_module {
  vm.func @min_i32_s(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.min.i32.s %arg0, %arg1 : i32
    %0 = vm.min.i32.s %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @max_i32_s
vm.module @my_module {
  vm.func @max_i32_s(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.max.i32.s %arg0, %arg1 : i32
    %0 = vm.max.i32.s %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

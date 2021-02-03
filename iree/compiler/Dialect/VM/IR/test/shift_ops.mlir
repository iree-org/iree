// Tests printing and parsing of bitwise shift and rotate ops.

// RUN: iree-opt -split-input-file %s | IreeFileCheck %s

// CHECK-LABEL: @shl_i32
vm.module @my_module {
  vm.func @shl_i32(%arg0 : i32) -> i32 {
    // CHECK: %0 = vm.shl.i32 %arg0, 2 : i32
    %0 = vm.shl.i32 %arg0, 2 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @shr_i32_s
vm.module @my_module {
  vm.func @shr_i32_s(%arg0 : i32) -> i32 {
    // CHECK: %0 = vm.shr.i32.s %arg0, 2 : i32
    %0 = vm.shr.i32.s %arg0, 2 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @shr_i32_u
vm.module @my_module {
  vm.func @shr_i32_u(%arg0 : i32) -> i32 {
    // CHECK: %0 = vm.shr.i32.u %arg0, 2 : i32
    %0 = vm.shr.i32.u %arg0, 2 : i32
    vm.return %0 : i32
  }
}

// RUN: iree-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: @shl_i32
vm.module @my_module {
  vm.func @shl_i32(%arg0 : i32) -> i32 {
    // CHECK: %[[C2:.+]] = vm.const.i32 2
    %c2 = vm.const.i32 2
    // CHECK: %0 = vm.shl.i32 %arg0, %[[C2]] : i32
    %0 = vm.shl.i32 %arg0, %c2 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @shr_i32_s
vm.module @my_module {
  vm.func @shr_i32_s(%arg0 : i32) -> i32 {
    // CHECK: %[[C2:.+]] = vm.const.i32 2
    %c2 = vm.const.i32 2
    // CHECK: %0 = vm.shr.i32.s %arg0, %[[C2]] : i32
    %0 = vm.shr.i32.s %arg0, %c2 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @shr_i32_u
vm.module @my_module {
  vm.func @shr_i32_u(%arg0 : i32) -> i32 {
    // CHECK: %[[C2:.+]] = vm.const.i32 2
    %c2 = vm.const.i32 2
    // CHECK: %0 = vm.shr.i32.u %arg0, %[[C2]] : i32
    %0 = vm.shr.i32.u %arg0, %c2 : i32
    vm.return %0 : i32
  }
}

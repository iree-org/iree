// Tests printing and parsing of casting/conversion ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @trunc
vm.module @my_module {
  vm.func @trunc(%arg0 : i32) -> i32 {
    // CHECK: %0 = vm.trunc.i8 %arg0 : i32
    %0 = vm.trunc.i8 %arg0 : i32
    // CHECK-NEXT: %1 = vm.trunc.i16 %0 : i32
    %1 = vm.trunc.i16 %0 : i32
    vm.return %1 : i32
  }
}

// -----

// CHECK-LABEL: @ext
vm.module @my_module {
  vm.func @ext(%arg0 : i32) -> i32 {
    // CHECK-NEXT: %0 = vm.ext.i8.i32.s %arg0 : i32
    %0 = vm.ext.i8.i32.s %arg0 : i32
    // CHECK-NEXT: %1 = vm.ext.i16.i32.s %0 : i32
    %1 = vm.ext.i16.i32.s %0 : i32
    vm.return %1 : i32
  }
}

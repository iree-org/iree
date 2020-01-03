// Tests printing and parsing of assignment ops.

// RUN: iree-opt -split-input-file %s | IreeFileCheck %s

// CHECK-LABEL: @select_i32
vm.module @my_module {
  vm.func @select_i32(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // CHECK: %0 = vm.select.i32 %arg0, %arg1, %arg2 : i32
    %0 = vm.select.i32 %arg0, %arg1, %arg2 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @select_ref
vm.module @my_module {
  vm.func @select_ref(%arg0 : i32,
                      %arg1 : !iree.opaque_ref,
                      %arg2 : !iree.opaque_ref) -> !iree.opaque_ref {
    // CHECK: %ref = vm.select.ref %arg0, %arg1, %arg2 : !iree.opaque_ref
    %ref = vm.select.ref %arg0, %arg1, %arg2 : !iree.opaque_ref
    vm.return %ref : !iree.opaque_ref
  }
}

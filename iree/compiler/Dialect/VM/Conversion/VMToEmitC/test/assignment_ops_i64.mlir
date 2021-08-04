// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-convert-vm-to-emitc)' %s | IreeFileCheck %s

// CHECK-LABEL: @my_module_select_i64
vm.module @my_module {
  vm.func @select_i64(%arg0 : i32, %arg1 : i64, %arg2 : i64) -> i64 {
    // CHECK: %0 = emitc.call "vm_select_i64"(%arg3, %arg4, %arg5) : (i32, i64, i64) -> i64
    %0 = vm.select.i64 %arg0, %arg1, %arg2 : i64
    vm.return %0 : i64
  }
}

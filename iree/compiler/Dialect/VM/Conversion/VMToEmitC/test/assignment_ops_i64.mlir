// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-convert-vm-to-emitc)' %s | IreeFileCheck %s

// CHECK-LABEL: vm.func @select_i64
vm.module @my_module {
  vm.func @select_i64(%arg0 : i32, %arg1 : i64, %arg2 : i64) -> i64 {
    // CHECK: %0 = emitc.call "vm_select_i64"(%arg0, %arg1, %arg2) {args = [0 : index, 1 : index, 2 : index]} : (i32, i64, i64) -> i64
    %0 = vm.select.i64 %arg0, %arg1, %arg2 : i64
    vm.return %0 : i64
  }
}

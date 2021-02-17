// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-convert-vm-to-emitc)' %s | IreeFileCheck %s

// CHECK-LABEL: vm.func @select_i32
vm.module @my_module {
  vm.func @select_i32(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // CHECK: %0 = emitc.call "vm_select_i32"(%arg0, %arg1, %arg2) {args = [0 : index, 1 : index, 2 : index]} : (i32, i32, i32) -> i32
    %0 = vm.select.i32 %arg0, %arg1, %arg2 : i32
    vm.return %0 : i32
  }
}

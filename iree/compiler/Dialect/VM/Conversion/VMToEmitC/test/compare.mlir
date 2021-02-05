// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-convert-vm-to-emitc)' %s | IreeFileCheck %s

// CHECK: vm.module @module {
vm.module @module {
  // CHECK-LABEL: vm.func @cmp_ne_i32
  vm.func @cmp_ne_i32(%arg0 : i32, %arg1 : i32) {
    // CHECK-NEXT: %0 = emitc.call "vm_cmp_ne_i32"(%arg0, %arg1) {args = [0 : index, 1 : index]} : (i32, i32) -> i32
    %0 = vm.cmp.ne.i32 %arg0, %arg1 : i32
    // CHECK-NEXT: vm.return
    vm.return
  }
}

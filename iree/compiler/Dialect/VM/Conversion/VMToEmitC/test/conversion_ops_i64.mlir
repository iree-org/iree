// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-convert-vm-to-emitc)' %s | IreeFileCheck %s

// CHECK-LABEL: vm.func @trunc_i64
vm.module @my_module {
  vm.func @trunc_i64(%arg0 : i64) -> i32 {
    // CHECK-NEXT: %0 = emitc.call "vm_trunc_i64i32"(%arg0) {args = [0 : index]} : (i64) -> i32
    %0 = vm.trunc.i64.i32 %arg0 : i64 -> i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: vm.func @ext_i64
vm.module @my_module {
  vm.func @ext_i64(%arg0 : i32) -> i64 {
    // CHECK-NEXT: %0 = emitc.call "vm_ext_i32i64s"(%arg0) {args = [0 : index]} : (i32) -> i64
    %0 = vm.ext.i32.i64.s %arg0 : i32 -> i64
    // CHECK-NEXT: %1 = emitc.call "vm_ext_i32i64u"(%arg0) {args = [0 : index]} : (i32) -> i64
    %1 = vm.ext.i32.i64.u %arg0 : i32 -> i64
    vm.return %1 : i64
  }
}

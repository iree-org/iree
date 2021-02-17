// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-convert-vm-to-emitc)' %s | IreeFileCheck %s

// CHECK-LABEL: @shl_i64
vm.module @my_module {
  vm.func @shl_i64(%arg0 : i64) -> i64 {
    // CHECK: %0 = emitc.call "vm_shl_i64"(%arg0) {args = [0 : index, 2 : i8]} : (i64) -> i64
    %0 = vm.shl.i64 %arg0, 2 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: @shr_i64_s
vm.module @my_module {
  vm.func @shr_i64_s(%arg0 : i64) -> i64 {
    // CHECK: %0 = emitc.call "vm_shr_i64s"(%arg0) {args = [0 : index, 2 : i8]} : (i64) -> i64
    %0 = vm.shr.i64.s %arg0, 2 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: @shr_i64_u
vm.module @my_module {
  vm.func @shr_i64_u(%arg0 : i64) -> i64 {
    // CHECK: %0 = emitc.call "vm_shr_i64u"(%arg0) {args = [0 : index, 2 : i8]} : (i64) -> i64
    %0 = vm.shr.i64.u %arg0, 2 : i64
    vm.return %0 : i64
  }
}

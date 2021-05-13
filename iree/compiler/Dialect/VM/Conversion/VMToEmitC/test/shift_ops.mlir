// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-convert-vm-to-emitc)' %s | IreeFileCheck %s

// CHECK-LABEL: @shl_i32
vm.module @my_module {
  vm.func @shl_i32(%arg0 : i32) -> i32 {
    %c2 = constant 2 : i32
    // CHECK: %0 = emitc.call "vm_shl_i32"(%arg0, %c2) {args = [0 : index, 1 : index]} : (i32) -> i32
    %0 = vm.shl.i32 %arg0, %c2 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @shr_i32_s
vm.module @my_module {
  vm.func @shr_i32_s(%arg0 : i32) -> i32 {
    %c2 = constant 2 : i32
    // CHECK: %0 = emitc.call "vm_shr_i32s"(%arg0, %c2) {args = [0 : index, 1 : index]} : (i32) -> i32
    %0 = vm.shr.i32.s %arg0, %c2 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @shr_i32_u
vm.module @my_module {
  vm.func @shr_i32_u(%arg0 : i32) -> i32 {
    %c2 = constant 2 : i32
    // CHECK: %0 = emitc.call "vm_shr_i32u"(%arg0, %c2) {args = [0 : index, 1 : index]} : (i32) -> i32
    %0 = vm.shr.i32.u %arg0, %c2 : i32
    vm.return %0 : i32
  }
}

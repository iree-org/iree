// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc)' %s | FileCheck %s

vm.module @module {
  // CHECK-LABEL: @module_cmp_eq_i32
  vm.func @cmp_eq_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK-NEXT: %0 = emitc.call "vm_cmp_eq_i32"(%arg3, %arg4) : (i32, i32) -> i32
    %0 = vm.cmp.eq.i32 %arg0, %arg1 : i32
    vm.return
  }
}

// -----

vm.module @module {
  // CHECK-LABEL: @module_cmp_ne_i32
  vm.func @cmp_ne_i32(%arg0 : i32, %arg1 : i32) {
    // CHECK-NEXT: %0 = emitc.call "vm_cmp_ne_i32"(%arg3, %arg4) : (i32, i32) -> i32
    %0 = vm.cmp.ne.i32 %arg0, %arg1 : i32
    vm.return
  }
}

// -----

vm.module @module {
  // CHECK-LABEL: @module_cmp_lt_i32_s
  vm.func @cmp_lt_i32_s(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK-NEXT: %0 = emitc.call "vm_cmp_lt_i32s"(%arg3, %arg4) : (i32, i32) -> i32
    %0 = vm.cmp.lt.i32.s %arg0, %arg1 : i32
    vm.return
  }
}

// -----

vm.module @module {
  // CHECK-LABEL: @module_cmp_lt_i32_u
  vm.func @cmp_lt_i32_u(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK-NEXT: %0 = emitc.call "vm_cmp_lt_i32u"(%arg3, %arg4) : (i32, i32) -> i32
    %0 = vm.cmp.lt.i32.u %arg0, %arg1 : i32
    vm.return
  }
}

// -----

vm.module @module {
  // CHECK-LABEL: @module_cmp_nz_i32
  vm.func @cmp_nz_i32(%arg0 : i32) -> i32 {
    // CHECK-NEXT: %0 = emitc.call "vm_cmp_nz_i32"(%arg3) : (i32) -> i32
    %0 = vm.cmp.nz.i32 %arg0 : i32
    vm.return
  }
}

// -----

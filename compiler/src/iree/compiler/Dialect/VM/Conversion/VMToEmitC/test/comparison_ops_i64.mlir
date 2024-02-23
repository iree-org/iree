// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

vm.module @module {
  // CHECK-LABEL: emitc.func private @module_cmp_eq_i64
  vm.func @cmp_eq_i64(%arg0 : i64, %arg1 : i64) -> i64 {
    // CHECK-NEXT: %0 = emitc.call_opaque "vm_cmp_eq_i64"(%arg3, %arg4) : (i64, i64) -> i32
    %0 = vm.cmp.eq.i64 %arg0, %arg1 : i64
    vm.return
  }
}

// -----

vm.module @module {
  // CHECK-LABEL: emitc.func private @module_cmp_ne_i64
  vm.func @cmp_ne_i64(%arg0 : i64, %arg1 : i64) {
    // CHECK-NEXT: %0 = emitc.call_opaque "vm_cmp_ne_i64"(%arg3, %arg4) : (i64, i64) -> i32
    %0 = vm.cmp.ne.i64 %arg0, %arg1 : i64
    vm.return
  }
}

// -----

vm.module @module {
  // CHECK-LABEL: emitc.func private @module_cmp_lt_i64_s
  vm.func @cmp_lt_i64_s(%arg0 : i64, %arg1 : i64) -> i64 {
    // CHECK-NEXT: %0 = emitc.call_opaque "vm_cmp_lt_i64s"(%arg3, %arg4) : (i64, i64) -> i32
    %0 = vm.cmp.lt.i64.s %arg0, %arg1 : i64
    vm.return
  }
}

// -----

vm.module @module {
  // CHECK-LABEL: emitc.func private @module_cmp_lt_i64_u
  vm.func @cmp_lt_i64_u(%arg0 : i64, %arg1 : i64) -> i64 {
    // CHECK-NEXT: %0 = emitc.call_opaque "vm_cmp_lt_i64u"(%arg3, %arg4) : (i64, i64) -> i32
    %0 = vm.cmp.lt.i64.u %arg0, %arg1 : i64
    vm.return
  }
}

// -----

vm.module @module {
  // CHECK-LABEL: emitc.func private @module_cmp_nz_i64
  vm.func @cmp_nz_i64(%arg0 : i64) -> i64 {
    // CHECK-NEXT: %0 = emitc.call_opaque "vm_cmp_nz_i64"(%arg3) : (i64) -> i32
    %0 = vm.cmp.nz.i64 %arg0 : i64
    vm.return
  }
}

// -----

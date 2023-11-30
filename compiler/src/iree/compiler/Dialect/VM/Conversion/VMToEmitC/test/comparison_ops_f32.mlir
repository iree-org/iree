// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

vm.module @module {
  // CHECK-LABEL: @module_cmp_eq_f32o
  vm.func @cmp_eq_f32o(%arg0 : f32, %arg1 : f32) -> i32 {
    // CHECK-NEXT: %0 = emitc.call_opaque "vm_cmp_eq_f32o"(%arg3, %arg4) : (f32, f32) -> i32
    %0 = vm.cmp.eq.f32.o %arg0, %arg1 : f32
    vm.return
  }
}

// -----

vm.module @module {
  // CHECK-LABEL: @module_cmp_eq_f32u
  vm.func @cmp_eq_f32u(%arg0 : f32, %arg1 : f32) -> i32 {
    // CHECK-NEXT: %0 = emitc.call_opaque "vm_cmp_eq_f32u"(%arg3, %arg4) : (f32, f32) -> i32
    %0 = vm.cmp.eq.f32.u %arg0, %arg1 : f32
    vm.return
  }
}

// -----

vm.module @module {
  // CHECK-LABEL: @module_cmp_ne_f32o
  vm.func @cmp_ne_f32o(%arg0 : f32, %arg1 : f32) -> i32 {
    // CHECK-NEXT: %0 = emitc.call_opaque "vm_cmp_ne_f32o"(%arg3, %arg4) : (f32, f32) -> i32
    %0 = vm.cmp.ne.f32.o %arg0, %arg1 : f32
    vm.return
  }
}

// -----

vm.module @module {
  // CHECK-LABEL: @module_cmp_ne_f32u
  vm.func @cmp_ne_f32u(%arg0 : f32, %arg1 : f32) -> i32 {
    // CHECK-NEXT: %0 = emitc.call_opaque "vm_cmp_ne_f32u"(%arg3, %arg4) : (f32, f32) -> i32
    %0 = vm.cmp.ne.f32.u %arg0, %arg1 : f32
    vm.return
  }
}

// -----

vm.module @module {
  // CHECK-LABEL: @module_cmp_lt_f32o
  vm.func @cmp_lt_f32o(%arg0 : f32, %arg1 : f32) -> i32 {
    // CHECK-NEXT: %0 = emitc.call_opaque "vm_cmp_lt_f32o"(%arg3, %arg4) : (f32, f32) -> i32
    %0 = vm.cmp.lt.f32.o %arg0, %arg1 : f32
    vm.return
  }
}

// -----

vm.module @module {
  // CHECK-LABEL: @module_cmp_lt_f32u
  vm.func @cmp_lt_f32u(%arg0 : f32, %arg1 : f32) -> i32 {
    // CHECK-NEXT: %0 = emitc.call_opaque "vm_cmp_lt_f32u"(%arg3, %arg4) : (f32, f32) -> i32
    %0 = vm.cmp.lt.f32.u %arg0, %arg1 : f32
    vm.return
  }
}

// -----

vm.module @module {
  // CHECK-LABEL: @module_cmp_lte_f32o
  vm.func @cmp_lte_f32o(%arg0 : f32, %arg1 : f32) -> i32 {
    // CHECK-NEXT: %0 = emitc.call_opaque "vm_cmp_lte_f32o"(%arg3, %arg4) : (f32, f32) -> i32
    %0 = vm.cmp.lte.f32.o %arg0, %arg1 : f32
    vm.return
  }
}

// -----

vm.module @module {
  // CHECK-LABEL: @module_cmp_lte_f32u
  vm.func @cmp_lte_f32u(%arg0 : f32, %arg1 : f32) -> i32 {
    // CHECK-NEXT: %0 = emitc.call_opaque "vm_cmp_lte_f32u"(%arg3, %arg4) : (f32, f32) -> i32
    %0 = vm.cmp.lte.f32.u %arg0, %arg1 : f32
    vm.return
  }
}

// -----

vm.module @module {
  // CHECK-LABEL: @module_cmp_nan_f32
  vm.func @cmp_nan_f32(%arg0 : f32) -> i32 {
    // CHECK-NEXT: %0 = emitc.call_opaque "vm_cmp_nan_f32"(%arg3) : (f32) -> i32
    %0 = vm.cmp.nan.f32 %arg0 : f32
    vm.return
  }
}

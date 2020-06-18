// Tests printing and parsing of comparison ops.

// RUN: iree-opt -split-input-file %s | IreeFileCheck %s

// CHECK-LABEL: @cmp_eq_i32
vm.module @my_module {
  vm.func @cmp_eq_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %eq = vm.cmp.eq.i32 %arg0, %arg1 : i32
    %eq = vm.cmp.eq.i32 %arg0, %arg1 : i32
    vm.return %eq : i32
  }
}

// -----

// CHECK-LABEL: @cmp_ne_i32
vm.module @my_module {
  vm.func @cmp_ne_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %ne = vm.cmp.ne.i32 %arg0, %arg1 : i32
    %ne = vm.cmp.ne.i32 %arg0, %arg1 : i32
    vm.return %ne : i32
  }
}

// -----

// CHECK-LABEL: @cmp_lt_i32_s
vm.module @my_module {
  vm.func @cmp_lt_i32_s(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %slt = vm.cmp.lt.i32.s %arg0, %arg1 : i32
    %slt = vm.cmp.lt.i32.s %arg0, %arg1 : i32
    vm.return %slt : i32
  }
}

// -----

// CHECK-LABEL: @cmp_lt_i32_u
vm.module @my_module {
  vm.func @cmp_lt_i32_u(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %ult = vm.cmp.lt.i32.u %arg0, %arg1 : i32
    %ult = vm.cmp.lt.i32.u %arg0, %arg1 : i32
    vm.return %ult : i32
  }
}

// -----

// CHECK-LABEL: @cmp_lte_i32_s
vm.module @my_module {
  vm.func @cmp_lte_i32_s(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %slte = vm.cmp.lte.i32.s %arg0, %arg1 : i32
    %slte = vm.cmp.lte.i32.s %arg0, %arg1 : i32
    vm.return %slte : i32
  }
}

// -----

// CHECK-LABEL: @cmp_lte_i32_u
vm.module @my_module {
  vm.func @cmp_lte_i32_u(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %ulte = vm.cmp.lte.i32.u %arg0, %arg1 : i32
    %ulte = vm.cmp.lte.i32.u %arg0, %arg1 : i32
    vm.return %ulte : i32
  }
}

// -----

// CHECK-LABEL: @cmp_gt_i32_s
vm.module @my_module {
  vm.func @cmp_gt_i32_s(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %sgt = vm.cmp.gt.i32.s %arg0, %arg1 : i32
    %sgt = vm.cmp.gt.i32.s %arg0, %arg1 : i32
    vm.return %sgt : i32
  }
}

// -----

// CHECK-LABEL: @cmp_gt_i32_u
vm.module @my_module {
  vm.func @cmp_gt_i32_u(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %ugt = vm.cmp.gt.i32.u %arg0, %arg1 : i32
    %ugt = vm.cmp.gt.i32.u %arg0, %arg1 : i32
    vm.return %ugt : i32
  }
}

// -----

// CHECK-LABEL: @cmp_gte_i32_s
vm.module @my_module {
  vm.func @cmp_gte_i32_s(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %sgte = vm.cmp.gte.i32.s %arg0, %arg1 : i32
    %sgte = vm.cmp.gte.i32.s %arg0, %arg1 : i32
    vm.return %sgte : i32
  }
}

// -----

// CHECK-LABEL: @cmp_gte_i32_u
vm.module @my_module {
  vm.func @cmp_gte_i32_u(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %ugte = vm.cmp.gte.i32.u %arg0, %arg1 : i32
    %ugte = vm.cmp.gte.i32.u %arg0, %arg1 : i32
    vm.return %ugte : i32
  }
}

// -----

// CHECK-LABEL: @cmp_nz_i32
vm.module @my_module {
  vm.func @cmp_nz_i32(%arg0 : i32) -> i32 {
    // CHECK: %nz = vm.cmp.nz.i32 %arg0 : i32
    %nz = vm.cmp.nz.i32 %arg0 : i32
    vm.return %nz : i32
  }
}

// -----

// CHECK-LABEL: @cmp_eq_ref
vm.module @my_module {
  vm.func @cmp_eq_ref(%arg0 : !vm.ref<?>,
                      %arg1 : !vm.ref<?>) -> i32 {
    // CHECK: %req = vm.cmp.eq.ref %arg0, %arg1 : !vm.ref<?>
    %req = vm.cmp.eq.ref %arg0, %arg1 : !vm.ref<?>
    vm.return %req : i32
  }
}

// -----

// CHECK-LABEL: @cmp_ne_ref
vm.module @my_module {
  vm.func @cmp_ne_ref(%arg0 : !vm.ref<?>,
                      %arg1 : !vm.ref<?>) -> i32 {
    // CHECK: %rne = vm.cmp.ne.ref %arg0, %arg1 : !vm.ref<?>
    %rne = vm.cmp.ne.ref %arg0, %arg1 : !vm.ref<?>
    vm.return %rne : i32
  }
}

// -----

// CHECK-LABEL: @cmp_nz_ref
vm.module @my_module {
  vm.func @cmp_nz_ref(%arg0 : !vm.ref<?>) -> i32 {
    // CHECK: %rnz = vm.cmp.nz.ref %arg0 : !vm.ref<?>
    %rnz = vm.cmp.nz.ref %arg0 : !vm.ref<?>
    vm.return %rnz : i32
  }
}

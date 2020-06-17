// Tests folding and canonicalization of comparison ops.

// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(canonicalize)' %s | IreeFileCheck %s

// CHECK-LABEL: @cmp_eq_i32_folds
vm.module @cmp_eq_i32_folds {
  // CHECK-LABEL: @always_eq
  vm.func @always_eq(%arg0 : i32) -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1 : i32
    %eq = vm.cmp.eq.i32 %arg0, %arg0 : i32
    vm.return %eq : i32
  }

  // CHECK-LABEL: @const_eq
  vm.func @const_eq() -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1 : i32
    %c1 = vm.const.i32 1 : i32
    %c1d = vm.const.i32 1 : i32
    %eq = vm.cmp.eq.i32 %c1, %c1d : i32
    vm.return %eq : i32
  }

  // CHECK-LABEL: @const_ne
  vm.func @const_ne() -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %c1 = vm.const.i32 1 : i32
    %c2 = vm.const.i32 2 : i32
    %eq = vm.cmp.eq.i32 %c1, %c2 : i32
    vm.return %eq : i32
  }
}

// -----

// CHECK-LABEL: @cmp_ne_i32_folds
vm.module @cmp_ne_i32_folds {
  // CHECK-LABEL: @always_eq
  vm.func @always_eq(%arg0 : i32) -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %ne = vm.cmp.ne.i32 %arg0, %arg0 : i32
    vm.return %ne : i32
  }

  // CHECK-LABEL: @always_ne
  vm.func @always_ne(%arg0 : i32, %arg1 : i32) -> i32 {
    // NOTE: do not fold, as can't be sure they are not equal.
    // CHECK: %ne = vm.cmp.ne.i32 %arg0, %arg1 : i32
    // CHECK-NEXT: vm.return %ne : i32
    %ne = vm.cmp.ne.i32 %arg0, %arg1 : i32
    vm.return %ne : i32
  }

  // CHECK-LABEL: @const_ne
  vm.func @const_ne() -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1 : i32
    %c1 = vm.const.i32 1 : i32
    %c2 = vm.const.i32 2 : i32
    %ne = vm.cmp.ne.i32 %c1, %c2 : i32
    vm.return %ne : i32
  }

  // CHECK-LABEL: @const_eq
  vm.func @const_eq() -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %c1 = vm.const.i32 1 : i32
    %c1d = vm.const.i32 1 : i32
    %ne = vm.cmp.ne.i32 %c1, %c1d : i32
    vm.return %ne : i32
  }

  // CHECK-LABEL: @cmp_const_zero
  vm.func @cmp_const_zero(%arg0 : i32, %arg1 : i32) -> (i32, i32) {
    // CHECK-DAG: vm.cmp.nz.i32 %arg0 : i32
    // CHECK-DAG: vm.cmp.nz.i32 %arg1 : i32
    %c0 = vm.const.i32 0 : i32
    %nz0 = vm.cmp.ne.i32 %arg0, %c0 : i32
    %nz1 = vm.cmp.ne.i32 %c0, %arg1 : i32
    vm.return %nz0, %nz1 : i32, i32
  }
}

// -----

// CHECK-LABEL: @cmp_slt_i32_folds
vm.module @cmp_slt_i32_folds {
  // CHECK-LABEL: @always_eq
  vm.func @always_eq(%arg0 : i32) -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %cmp = vm.cmp.lt.i32.s %arg0, %arg0 : i32
    vm.return %cmp : i32
  }

  // CHECK-LABEL: @const_true
  vm.func @const_true() -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1 : i32
    %c1 = vm.const.i32 -1 : i32
    %c2 = vm.const.i32 2 : i32
    %cmp = vm.cmp.lt.i32.s %c1, %c2 : i32
    vm.return %cmp : i32
  }

  // CHECK-LABEL: @const_false
  vm.func @const_false() -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %c1 = vm.const.i32 -1 : i32
    %c2 = vm.const.i32 2 : i32
    %cmp = vm.cmp.lt.i32.s %c2, %c1 : i32
    vm.return %cmp : i32
  }
}

// -----

// CHECK-LABEL: @cmp_ult_i32_folds
vm.module @cmp_ult_i32_folds {
  // CHECK-LABEL: @always_eq
  vm.func @always_eq(%arg0 : i32) -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %cmp = vm.cmp.lt.i32.u %arg0, %arg0 : i32
    vm.return %cmp : i32
  }

  // CHECK-LABEL: @const_true
  vm.func @const_true() -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1 : i32
    %c1 = vm.const.i32 -1 : i32
    %c2 = vm.const.i32 2 : i32
    %cmp = vm.cmp.lt.i32.u %c2, %c1 : i32
    vm.return %cmp : i32
  }

  // CHECK-LABEL: @const_false
  vm.func @const_false() -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %c1 = vm.const.i32 -1 : i32
    %c2 = vm.const.i32 2 : i32
    %cmp = vm.cmp.lt.i32.u %c1, %c2 : i32
    vm.return %cmp : i32
  }
}

// -----

// CHECK-LABEL: @cmp_slte_i32_folds
vm.module @cmp_slte_i32_folds {
  // CHECK-LABEL: @always_eq
  vm.func @always_eq(%arg0 : i32) -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1 : i32
    %cmp = vm.cmp.lte.i32.s %arg0, %arg0 : i32
    vm.return %cmp : i32
  }

  // CHECK-LABEL: @const_true
  vm.func @const_true() -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1 : i32
    %c1 = vm.const.i32 -1 : i32
    %c2 = vm.const.i32 2 : i32
    %cmp = vm.cmp.lte.i32.s %c1, %c2 : i32
    vm.return %cmp : i32
  }

  // CHECK-LABEL: @const_false
  vm.func @const_false() -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %c1 = vm.const.i32 -1 : i32
    %c2 = vm.const.i32 2 : i32
    %cmp = vm.cmp.lte.i32.s %c2, %c1 : i32
    vm.return %cmp : i32
  }
}

// -----

// CHECK-LABEL: @cmp_ulte_i32_folds
vm.module @cmp_ulte_i32_folds {
  // CHECK-LABEL: @always_eq
  vm.func @always_eq(%arg0 : i32) -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1 : i32
    %cmp = vm.cmp.lte.i32.u %arg0, %arg0 : i32
    vm.return %cmp : i32
  }

  // CHECK-LABEL: @const_true
  vm.func @const_true() -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1 : i32
    %c1 = vm.const.i32 -1 : i32
    %c2 = vm.const.i32 2 : i32
    %cmp = vm.cmp.lte.i32.u %c2, %c1 : i32
    vm.return %cmp : i32
  }

  // CHECK-LABEL: @const_false
  vm.func @const_false() -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %c1 = vm.const.i32 -1 : i32
    %c2 = vm.const.i32 2 : i32
    %cmp = vm.cmp.lte.i32.u %c1, %c2 : i32
    vm.return %cmp : i32
  }
}

// -----

// CHECK-LABEL: @cmp_sgt_i32_folds
vm.module @cmp_sgt_i32_folds {
  // CHECK-LABEL: @always_eq
  vm.func @always_eq(%arg0 : i32) -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %cmp = vm.cmp.gt.i32.s %arg0, %arg0 : i32
    vm.return %cmp : i32
  }

  // CHECK-LABEL: @const_true
  vm.func @const_true() -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1 : i32
    %c1 = vm.const.i32 -1 : i32
    %c2 = vm.const.i32 2 : i32
    %cmp = vm.cmp.gt.i32.s %c2, %c1 : i32
    vm.return %cmp : i32
  }

  // CHECK-LABEL: @const_false
  vm.func @const_false() -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %c1 = vm.const.i32 -1 : i32
    %c2 = vm.const.i32 2 : i32
    %cmp = vm.cmp.gt.i32.s %c1, %c2 : i32
    vm.return %cmp : i32
  }
}

// -----

// CHECK-LABEL: @cmp_ugt_i32_folds
vm.module @cmp_ugt_i32_folds {
  // CHECK-LABEL: @always_eq
  vm.func @always_eq(%arg0 : i32) -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %cmp = vm.cmp.gt.i32.u %arg0, %arg0 : i32
    vm.return %cmp : i32
  }

  // CHECK-LABEL: @const_true
  vm.func @const_true() -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1 : i32
    %c1 = vm.const.i32 -1 : i32
    %c2 = vm.const.i32 2 : i32
    %cmp = vm.cmp.gt.i32.u %c1, %c2 : i32
    vm.return %cmp : i32
  }

  // CHECK-LABEL: @const_false
  vm.func @const_false() -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %c1 = vm.const.i32 -1 : i32
    %c2 = vm.const.i32 2 : i32
    %cmp = vm.cmp.gt.i32.u %c2, %c1 : i32
    vm.return %cmp : i32
  }
}

// -----

// CHECK-LABEL: @cmp_sgte_i32_folds
vm.module @cmp_sgte_i32_folds {
  // CHECK-LABEL: @always_eq
  vm.func @always_eq(%arg0 : i32) -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1 : i32
    %cmp = vm.cmp.gte.i32.s %arg0, %arg0 : i32
    vm.return %cmp : i32
  }

  // CHECK-LABEL: @const_true
  vm.func @const_true() -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1 : i32
    %c1 = vm.const.i32 -1 : i32
    %c2 = vm.const.i32 2 : i32
    %cmp = vm.cmp.gte.i32.s %c2, %c1 : i32
    vm.return %cmp : i32
  }

  // CHECK-LABEL: @const_false
  vm.func @const_false() -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %c1 = vm.const.i32 -1 : i32
    %c2 = vm.const.i32 2 : i32
    %cmp = vm.cmp.gte.i32.s %c1, %c2 : i32
    vm.return %cmp : i32
  }
}

// -----

// CHECK-LABEL: @cmp_ugte_i32_folds
vm.module @cmp_ugte_i32_folds {
  // CHECK-LABEL: @always_eq
  vm.func @always_eq(%arg0 : i32) -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1 : i32
    %cmp = vm.cmp.gte.i32.u %arg0, %arg0 : i32
    vm.return %cmp : i32
  }

  // CHECK-LABEL: @const_true
  vm.func @const_true() -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1 : i32
    %c1 = vm.const.i32 -1 : i32
    %c2 = vm.const.i32 2 : i32
    %cmp = vm.cmp.gte.i32.u %c1, %c2 : i32
    vm.return %cmp : i32
  }

  // CHECK-LABEL: @const_false
  vm.func @const_false() -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %c1 = vm.const.i32 -1 : i32
    %c2 = vm.const.i32 2 : i32
    %cmp = vm.cmp.gte.i32.u %c2, %c1 : i32
    vm.return %cmp : i32
  }
}

// -----

// CHECK-LABEL: @cmp_nz_i32_folds
vm.module @cmp_nz_i32_folds {
  // CHECK-LABEL: @const_nonzero
  vm.func @const_nonzero() -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1 : i32
    %c1 = vm.const.i32 1 : i32
    %nz = vm.cmp.nz.i32 %c1 : i32
    vm.return %nz : i32
  }

  // CHECK-LABEL: @const_zero
  vm.func @const_zero() -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %c0 = vm.const.i32 0 : i32
    %nz = vm.cmp.nz.i32 %c0 : i32
    vm.return %nz : i32
  }
}

// -----

// CHECK-LABEL: @cmp_eq_ref_folds
vm.module @cmp_eq_ref_folds {
  // CHECK-LABEL: @always_eq
  vm.func @always_eq(%arg0 : !vm.ref<?>) -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1 : i32
    %eq = vm.cmp.eq.ref %arg0, %arg0 : !vm.ref<?>
    vm.return %eq : i32
  }

  // CHECK-LABEL: @const_eq
  vm.func @const_eq() -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1 : i32
    %null = vm.const.ref.zero : !vm.ref<?>
    %null0 = vm.const.ref.zero : !vm.ref<?>
    %eq = vm.cmp.eq.ref %null, %null0 : !vm.ref<?>
    vm.return %eq : i32
  }

  // CHECK-LABEL: @cmp_null
  vm.func @cmp_null(%arg0 : !vm.ref<?>) -> i32 {
    // CHECK: %rnz = vm.cmp.nz.ref %arg0 : !vm.ref<?>
    // CHECK-NEXT: %0 = vm.xor.i32 %rnz, %c1 : i32
    // CHECK-NEXT: vm.return %0 : i32
    %null = vm.const.ref.zero : !vm.ref<?>
    %eq = vm.cmp.eq.ref %arg0, %null : !vm.ref<?>
    vm.return %eq : i32
  }
}

// -----

// CHECK-LABEL: @cmp_ne_ref_folds
vm.module @cmp_ne_ref_folds {
  // CHECK-LABEL: @always_eq
  vm.func @always_eq(%arg0 :  !vm.ref<?>) -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %ne = vm.cmp.ne.ref %arg0, %arg0 :  !vm.ref<?>
    vm.return %ne : i32
  }

  // CHECK-LABEL: @const_eq
  vm.func @const_eq() -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %null = vm.const.ref.zero : !vm.ref<?>
    %null0 = vm.const.ref.zero : !vm.ref<?>
    %ne = vm.cmp.ne.ref %null, %null0 : !vm.ref<?>
    vm.return %ne : i32
  }

  // CHECK-LABEL: @cmp_null
  vm.func @cmp_null(%arg0 : !vm.ref<?>) -> i32 {
    // CHECK: %rnz = vm.cmp.nz.ref %arg0 : !vm.ref<?>
    // CHECK-NEXT: vm.return %rnz : i32
    %null = vm.const.ref.zero : !vm.ref<?>
    %ne = vm.cmp.ne.ref %arg0, %null : !vm.ref<?>
    vm.return %ne : i32
  }
}

// -----

// CHECK-LABEL: @cmp_nz_ref_folds
vm.module @cmp_nz_ref_folds {
  // CHECK-LABEL: @const_null
  vm.func @const_null() -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %null = vm.const.ref.zero : !vm.ref<?>
    %ne = vm.cmp.nz.ref %null : !vm.ref<?>
    vm.return %ne : i32
  }
}

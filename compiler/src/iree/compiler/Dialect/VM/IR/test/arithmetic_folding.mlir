// Tests folding and canonicalization of arithmetic ops.

// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(canonicalize))" %s | FileCheck %s

// CHECK-LABEL: @add_i32_folds
vm.module @add_i32_folds {
  // CHECK-LABEL: @add_i32_0_y
  vm.func @add_i32_0_y(%arg0: i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %zero = vm.const.i32.zero
    %0 = vm.add.i32 %zero, %arg0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @add_i32_x_0
  vm.func @add_i32_x_0(%arg0: i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %zero = vm.const.i32.zero
    %0 = vm.add.i32 %arg0, %zero : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @add_i32_const
  vm.func @add_i32_const() -> i32 {
    // CHECK: %c5 = vm.const.i32 5
    // CHECK-NEXT: vm.return %c5 : i32
    %c1 = vm.const.i32 1
    %c4 = vm.const.i32 4
    %0 = vm.add.i32 %c1, %c4 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @mul_add_i32_lhs
  vm.func @mul_add_i32_lhs(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
    %0 = vm.mul.i32 %arg0, %arg1 : i32
    // CHECK: %[[RET:.+]] = vm.fma.i32 %arg0, %arg1, %arg2 : i32
    %1 = vm.add.i32 %0, %arg2 : i32
    // CHECK: return %[[RET]]
    vm.return %1 : i32
  }

  // CHECK-LABEL: @mul_add_i32_rhs
  vm.func @mul_add_i32_rhs(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
    %0 = vm.mul.i32 %arg0, %arg1 : i32
    // CHECK: %[[RET:.+]] = vm.fma.i32 %arg0, %arg1, %arg2 : i32
    %1 = vm.add.i32 %arg2, %0 : i32
    // CHECK: return %[[RET]]
    vm.return %1 : i32
  }

  // Expect this not to fold:
  // CHECK-LABEL: @mul_add_i32_multiple_users
  vm.func @mul_add_i32_multiple_users(%arg0: i32, %arg1: i32, %arg2: i32) -> (i32, i32) {
    // CHECK: vm.mul.i32
    %0 = vm.mul.i32 %arg0, %arg1 : i32
    // CHECK-NOT: vm.fma.i32
    // CHECK-NEXT: vm.add.i32
    %1 = vm.add.i32 %0, %arg2 : i32
    // CHECK-NEXT: vm.add.i32
    %2 = vm.add.i32 %0, %arg1 : i32
    vm.return %1, %2 : i32, i32
  }

  // Expect this not to fold:
  // CHECK-LABEL: @mul_add_i32_dont_sink
  vm.func @mul_add_i32_dont_sink(%arg0: i32, %arg1: i32, %arg2: i32, %cond: i32) -> i32 {
    // CHECK: vm.mul.i32
    %0 = vm.mul.i32 %arg0, %arg1 : i32
    vm.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    // CHECK: vm.add.i32
    %1 = vm.add.i32 %0, %arg2 : i32
    vm.return %1 : i32
  ^bb2:
    // CHECK: vm.add.i32
    %2 = vm.add.i32 %0, %arg1 : i32
    // CHECK: vm.div.i32.s
    %3 = vm.div.i32.s %2, %arg1 : i32
    vm.return %3 : i32
  }
}

// -----

// CHECK-LABEL: @sub_i32_folds
vm.module @sub_i32_folds {
  // CHECK-LABEL: @sub_i32_x_0
  vm.func @sub_i32_x_0(%arg0: i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %zero = vm.const.i32.zero
    %0 = vm.sub.i32 %arg0, %zero : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @sub_i32_const
  vm.func @sub_i32_const() -> i32 {
    // CHECK: %c3 = vm.const.i32 3
    // CHECK-NEXT: vm.return %c3 : i32
    %c1 = vm.const.i32 1
    %c4 = vm.const.i32 4
    %0 = vm.sub.i32 %c4, %c1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @add_sub_i32_folds
vm.module @add_sub_i32_folds {
  // CHECK-LABEL: @add_sub_x
  vm.func @add_sub_x(%arg0: i32, %arg1: i32) -> i32 {
    // CHECK-NEXT: vm.return %arg0
    %0 = vm.add.i32 %arg0, %arg1 : i32
    %1 = vm.sub.i32 %0, %arg1 : i32
    vm.return %1 : i32
  }
  // CHECK-LABEL: @add_sub_x_rev
  vm.func @add_sub_x_rev(%arg0: i32, %arg1: i32) -> i32 {
    // CHECK-NEXT: vm.return %arg0
    %0 = vm.add.i32 %arg1, %arg0 : i32
    %1 = vm.sub.i32 %arg1, %0 : i32
    vm.return %1 : i32
  }

  // CHECK-LABEL: @sub_add_x
  vm.func @sub_add_x(%arg0: i32, %arg1: i32) -> i32 {
    // CHECK-NEXT: vm.return %arg0
    %0 = vm.sub.i32 %arg0, %arg1 : i32
    %1 = vm.add.i32 %0, %arg1 : i32
    vm.return %1 : i32
  }
  // CHECK-LABEL: @sub_add_x_rev
  vm.func @sub_add_x_rev(%arg0: i32, %arg1: i32) -> i32 {
    // CHECK-NEXT: vm.return %arg0
    %0 = vm.sub.i32 %arg0, %arg1 : i32
    %1 = vm.add.i32 %arg1, %0 : i32
    vm.return %1 : i32
  }
}

// -----

// CHECK-LABEL: @mul_i32_folds
vm.module @mul_i32_folds {
  // CHECK-LABEL: @mul_i32_by_0
  vm.func @mul_i32_by_0(%arg0: i32) -> i32 {
    // CHECK: %zero = vm.const.i32.zero
    // CHECK-NEXT: vm.return %zero : i32
    %zero = vm.const.i32.zero
    %0 = vm.mul.i32 %arg0, %zero : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @mul_i32_1_y
  vm.func @mul_i32_1_y(%arg0: i32) -> i32 {
    // CHECK-NEXT: vm.return %arg0 : i32
    %c1 = vm.const.i32 1
    %0 = vm.mul.i32 %c1, %arg0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @mul_i32_x_1
  vm.func @mul_i32_x_1(%arg0: i32) -> i32 {
    // CHECK-NEXT: vm.return %arg0 : i32
    %c1 = vm.const.i32 1
    %0 = vm.mul.i32 %arg0, %c1 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @mul_i32_const
  vm.func @mul_i32_const() -> i32 {
    // CHECK: %c8 = vm.const.i32 8
    // CHECK-NEXT: vm.return %c8 : i32
    %c2 = vm.const.i32 2
    %c4 = vm.const.i32 4
    %0 = vm.mul.i32 %c2, %c4 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @mul_mul_i32_folds
vm.module @mul_mul_i32_folds {
  // CHECK-LABEL: @mul_mul_i32_const
  vm.func @mul_mul_i32_const(%arg0: i32) -> i32 {
    // CHECK: %c40 = vm.const.i32 40
    %c4 = vm.const.i32 4
    %c10 = vm.const.i32 10
    // CHECK: %0 = vm.mul.i32 %arg0, %c40 : i32
    %0 = vm.mul.i32 %arg0, %c4 : i32
    %1 = vm.mul.i32 %0, %c10 : i32
    // CHECK-NEXT: vm.return %0 : i32
    vm.return %1 : i32
  }
}

// -----

// CHECK-LABEL: @div_i32_folds
vm.module @div_i32_folds {
  // CHECK-LABEL: @div_i32_0_y
  vm.func @div_i32_0_y(%arg0: i32) -> i32 {
    // CHECK: %zero = vm.const.i32.zero
    // CHECK-NEXT: vm.return %zero : i32
    %zero = vm.const.i32.zero
    %0 = vm.div.i32.s %zero, %arg0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @div_i32_x_1
  vm.func @div_i32_x_1(%arg0: i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %c1 = vm.const.i32 1
    %0 = vm.div.i32.s %arg0, %c1 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @div_i32_const
  vm.func @div_i32_const() -> i32 {
    // CHECK: %c3 = vm.const.i32 3
    // CHECK-NEXT: vm.return %c3 : i32
    %c15 = vm.const.i32 15
    %c5 = vm.const.i32 5
    %0 = vm.div.i32.s %c15, %c5 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @mul_div_i32
  vm.func @mul_div_i32(%arg0: i32, %arg1: i32) -> i32 {
    %0 = vm.mul.i32 %arg0, %arg1 : i32
    %1 = vm.div.i32.s %0, %arg1 : i32
    // CHECK-NEXT: vm.return %arg0
    vm.return %1 : i32
  }
}

// -----

// CHECK-LABEL: @rem_i32_folds
vm.module @rem_i32_folds {
  // CHECK-LABEL: @rem_i32_x_1
  vm.func @rem_i32_x_1(%arg0: i32) -> i32 {
    // CHECK: %zero = vm.const.i32.zero
    // CHECK-NEXT: vm.return %zero : i32
    %c1 = vm.const.i32 1
    %0 = vm.rem.i32.s %arg0, %c1 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @rem_i32_0_y
  vm.func @rem_i32_0_y(%arg0: i32) -> i32 {
    // CHECK: %zero = vm.const.i32.zero
    // CHECK-NEXT: vm.return %zero : i32
    %zero = vm.const.i32.zero
    %0 = vm.rem.i32.s %zero, %arg0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @rem_i32_const
  vm.func @rem_i32_const() -> i32 {
    // CHECK: %c1 = vm.const.i32 1
    // CHECK-NEXT: vm.return %c1 : i32
    %c3 = vm.const.i32 3
    %c4 = vm.const.i32 4
    %0 = vm.rem.i32.s %c4, %c3 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @abs_i32_folds
vm.module @abs_i32_folds {
  // CHECK-LABEL: @abs_i32_const
  vm.func @abs_i32_const() -> i32 {
    // CHECK: %c2 = vm.const.i32 2
    // CHECK-NEXT: vm.return %c2 : i32
    %cn2 = vm.const.i32 -2
    %0 = vm.abs.i32 %cn2 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @min_i32_folds
vm.module @min_i32_folds {
  // CHECK-LABEL: @min_i32_cst_y
  vm.func @min_i32_cst_y(%arg0: i32) -> i32 {
    %c123 = vm.const.i32 123
    // CHECK: vm.min.i32.s %arg0, %c123 : i32
    %0 = vm.min.i32.s %c123, %arg0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @min_i32_x_x
  vm.func @min_i32_x_x(%arg0: i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %0 = vm.min.i32.s %arg0, %arg0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @min_i32_s_const
  vm.func @min_i32_s_const() -> i32 {
    // CHECK: %[[CST:.+]] = vm.const.i32 -5
    // CHECK-NEXT: vm.return %[[CST]] : i32
    %cn5 = vm.const.i32 -5
    %c4 = vm.const.i32 4
    %0 = vm.min.i32.s %cn5, %c4 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @min_i32_u_const
  vm.func @min_i32_u_const() -> i32 {
    // CHECK: %[[CST:.+]] = vm.const.i32 2147483647
    // CHECK-NEXT: vm.return %[[CST]] : i32
    %c7f = vm.const.i32 0x7FFFFFFF
    %c80 = vm.const.i32 0x80000000
    %0 = vm.min.i32.u %c7f, %c80 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @not_i32_folds
vm.module @not_i32_folds {
  // CHECK-LABEL: @not_i32_const
  vm.func @not_i32_const() -> i32 {
    // CHECK: %c889262066 = vm.const.i32 889262066
    // CHECK-NEXT: vm.return %c889262066 : i32
    %c = vm.const.i32 0xCAFEF00D
    %0 = vm.not.i32 %c : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @and_i32_folds
vm.module @and_i32_folds {
  // CHECK-LABEL: @and_i32_zero
  vm.func @and_i32_zero(%arg0: i32) -> i32 {
    // CHECK: %zero = vm.const.i32.zero
    // CHECK-NEXT: vm.return %zero : i32
    %zero = vm.const.i32.zero
    %0 = vm.and.i32 %arg0, %zero : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @and_i32_eq
  vm.func @and_i32_eq(%arg0: i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %0 = vm.and.i32 %arg0, %arg0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @and_i32_const
  vm.func @and_i32_const() -> i32 {
    // CHECK: %c1 = vm.const.i32 1
    // CHECK-NEXT: vm.return %c1 : i32
    %c1 = vm.const.i32 1
    %c3 = vm.const.i32 3
    %0 = vm.and.i32 %c1, %c3 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @or_i32_folds
vm.module @or_i32_folds {
  // CHECK-LABEL: @or_i32_0_y
  vm.func @or_i32_0_y(%arg0: i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %zero = vm.const.i32.zero
    %0 = vm.or.i32 %zero, %arg0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @or_i32_x_0
  vm.func @or_i32_x_0(%arg0: i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %zero = vm.const.i32.zero
    %0 = vm.or.i32 %arg0, %zero : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @or_i32_x_x
  vm.func @or_i32_x_x(%arg0: i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %0 = vm.or.i32 %arg0, %arg0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @or_i32_const
  vm.func @or_i32_const() -> i32 {
    // CHECK: %c5 = vm.const.i32 5
    // CHECK-NEXT: vm.return %c5 : i32
    %c1 = vm.const.i32 1
    %c4 = vm.const.i32 4
    %0 = vm.or.i32 %c1, %c4 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @xor_i32_folds
vm.module @xor_i32_folds {
  // CHECK-LABEL: @xor_i32_0_y
  vm.func @xor_i32_0_y(%arg0: i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %zero = vm.const.i32.zero
    %0 = vm.xor.i32 %zero, %arg0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @xor_i32_x_0
  vm.func @xor_i32_x_0(%arg0: i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %zero = vm.const.i32.zero
    %0 = vm.xor.i32 %arg0, %zero : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @xor_i32_x_x
  vm.func @xor_i32_x_x(%arg0: i32) -> i32 {
    // CHECK: %zero = vm.const.i32.zero
    // CHECK-NEXT: vm.return %zero : i32
    %0 = vm.xor.i32 %arg0, %arg0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @xor_i32_const
  vm.func @xor_i32_const() -> i32 {
    // CHECK: %c2 = vm.const.i32 2
    // CHECK-NEXT: vm.return %c2 : i32
    %c1 = vm.const.i32 1
    %c3 = vm.const.i32 3
    %0 = vm.xor.i32 %c1, %c3 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @ctlz_i32_folds
vm.module @ctlz_i32_folds {
  // CHECK-LABEL: @ctlz_i32_const_zero
  vm.func @ctlz_i32_const_zero() -> i32 {
    %c = vm.const.i32 0
    %0 = vm.ctlz.i32 %c : i32
    // CHECK: vm.return %c32 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @ctlz_i32_const_1
  vm.func @ctlz_i32_const_1() -> i32 {
    %c = vm.const.i32 1
    %0 = vm.ctlz.i32 %c : i32
    // CHECK: vm.return %c31 : i32
    vm.return %0 : i32
  }


  // CHECK-LABEL: @ctlz_i32_const_ffffffff
  vm.func @ctlz_i32_const_ffffffff() -> i32 {
    %c = vm.const.i32 0xFFFFFFFF
    %0 = vm.ctlz.i32 %c : i32
    // CHECK: vm.return %zero : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @shl_i32_folds
vm.module @shl_i32_folds {
  // CHECK-LABEL: @shl_i32_0_by_y
  vm.func @shl_i32_0_by_y() -> i32 {
    // CHECK: %zero = vm.const.i32.zero
    // CHECK-NEXT: vm.return %zero : i32
    %zero = vm.const.i32.zero
    %c4 = vm.const.i32 4
    %0 = vm.shl.i32 %zero, %c4 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @shl_i32_x_by_0
  vm.func @shl_i32_x_by_0(%arg0: i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %c0 = vm.const.i32 0
    %0 = vm.shl.i32 %arg0, %c0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @shl_i32_const
  vm.func @shl_i32_const() -> i32 {
    // CHECK: %c16 = vm.const.i32 16
    // CHECK-NEXT: vm.return %c16 : i32
    %c1 = vm.const.i32 1
    %c4 = vm.const.i32 4
    %0 = vm.shl.i32 %c1, %c4 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @shr_i32_s_folds
vm.module @shr_i32_s_folds {
  // CHECK-LABEL: @shr_i32_s_0_by_y
  vm.func @shr_i32_s_0_by_y() -> i32 {
    // CHECK: %zero = vm.const.i32.zero
    // CHECK-NEXT: vm.return %zero : i32
    %c0 = vm.const.i32.zero
    %c4 = vm.const.i32 4
    %0 = vm.shr.i32.s %c0, %c4 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @shr_i32_s_x_by_0
  vm.func @shr_i32_s_x_by_0(%arg0: i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %c0 = vm.const.i32.zero
    %0 = vm.shr.i32.s %arg0, %c0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @shr_i32_s_const
  vm.func @shr_i32_s_const() -> i32 {
    // CHECK: %[[C:.+]] = vm.const.i32 -134217728
    // CHECK-NEXT: vm.return %[[C]]
    %c = vm.const.i32 0x80000000
    %c4 = vm.const.i32 4
    %0 = vm.shr.i32.s %c, %c4 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @shr_i32_u_folds
vm.module @shr_i32_u_folds {
  // CHECK-LABEL: @shr_i32_u_0_by_y
  vm.func @shr_i32_u_0_by_y() -> i32 {
    // CHECK: %zero = vm.const.i32.zero
    // CHECK-NEXT: vm.return %zero : i32
    %zero = vm.const.i32.zero
    %c4 = vm.const.i32 4
    %0 = vm.shr.i32.u %zero, %c4 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @shr_i32_u_x_by_0
  vm.func @shr_i32_u_x_by_0(%arg0: i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %c0 = vm.const.i32 0
    %0 = vm.shr.i32.u %arg0, %c0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @shr_i32_u_const
  vm.func @shr_i32_u_const() -> i32 {
    // CHECK: %[[C:.+]] = vm.const.i32 134217728
    // CHECK-NEXT: vm.return %[[C]]
    %c = vm.const.i32 0x80000000
    %c4 = vm.const.i32 4
    %0 = vm.shr.i32.u %c, %c4 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @shr_i64_u_folds
vm.module @shr_i64_u_folds {
  // CHECK-LABEL: @shr_i64_u_0_by_y
  vm.func @shr_i64_u_0_by_y() -> i64 {
    // CHECK: %zero = vm.const.i64.zero
    // CHECK-NEXT: vm.return %zero : i64
    %zero = vm.const.i64.zero
    %c4 = vm.const.i32 4
    %0 = vm.shr.i64.u %zero, %c4 : i64
    vm.return %0 : i64
  }

  // CHECK-LABEL: @shr_i64_u_x_by_0
  vm.func @shr_i64_u_x_by_0(%arg0 : i64) -> i64 {
    // CHECK: vm.return %arg0 : i64
    %c0 = vm.const.i32 0
    %0 = vm.shr.i64.u %arg0, %c0 : i64
    vm.return %0 : i64
  }

  // CHECK-LABEL: @shr_i64_u_const
  vm.func @shr_i64_u_const() -> i64 {
    // CHECK: %[[C:.+]] = vm.const.i64 576460752303423488
    // CHECK-NEXT: vm.return %[[C]]
    %c = vm.const.i64 0x8000000000000000
    %c4 = vm.const.i32 4
    %0 = vm.shr.i64.u %c, %c4 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: @fma_i32_folds
vm.module @fma_i32_folds {
  // CHECK-LABEL: @fma_i32_0_b_c
  vm.func @fma_i32_0_b_c(%b: i32, %c: i32) -> i32 {
    %c0 = vm.const.i32.zero
    %d = vm.fma.i32 %c0, %b, %c : i32
    vm.return %d : i32
  }

  // CHECK-LABEL: @fma_i32_1_b_c
  vm.func @fma_i32_1_b_c(%b: i32, %c: i32) -> i32 {
    %c1 = vm.const.i32 1
    %d = vm.fma.i32 %c1, %b, %c : i32
    vm.return %d : i32
  }

  // CHECK-LABEL: @fma_i32_a_1_c
  vm.func @fma_i32_a_1_c(%a: i32, %c: i32) -> i32 {
    %c1 = vm.const.i32 1
    %d = vm.fma.i32 %a, %c1, %c : i32
    vm.return %d : i32
  }

  // CHECK-LABEL: @fma_i32_a_b_0
  vm.func @fma_i32_a_b_0(%a: i32, %b: i32) -> i32 {
    %c0 = vm.const.i32.zero
    %d = vm.fma.i32 %a, %b, %c0 : i32
    vm.return %d : i32
  }
}

// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(test-iree-convert-std-to-vm)" %s | FileCheck %s

// -----
// CHECK-LABEL: @t001_cmp_eq_i32
module @t001_cmp_eq_i32 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.eq.i32 %[[ARG0]], %[[ARG1]] : i32
    %1 = arith.cmpi eq, %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t002_cmp_ne_i32
module @t002_cmp_ne_i32 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.ne.i32 %[[ARG0]], %[[ARG1]] : i32
    %1 = arith.cmpi ne, %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t003_cmp_slt_i32
module @t003_cmp_slt_i32 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.lt.i32.s %[[ARG0]], %[[ARG1]] : i32
    %1 = arith.cmpi slt, %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t004_cmp_sle_i32
module @t004_cmp_sle_i32 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.lte.i32.s %[[ARG0]], %[[ARG1]] : i32
    %1 = arith.cmpi sle, %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t005_cmp_sgt_i32
module @t005_cmp_sgt_i32 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.gt.i32.s %[[ARG0]], %[[ARG1]] : i32
    %1 = arith.cmpi sgt, %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t006_cmp_sge_i32
module @t006_cmp_sge_i32 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.gte.i32.s %[[ARG0]], %[[ARG1]] : i32
    %1 = arith.cmpi sge, %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t007_cmp_ult_i32
module @t007_cmp_ult_i32 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.lt.i32.u %[[ARG0]], %[[ARG1]] : i32
    %1 = arith.cmpi ult, %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t008_cmp_ule_i32
module @t008_cmp_ule_i32 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.lte.i32.u %[[ARG0]], %[[ARG1]] : i32
    %1 = arith.cmpi ule, %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t009_cmp_ugt_i32
module @t009_cmp_ugt_i32 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.gt.i32.u %[[ARG0]], %[[ARG1]] : i32
    %1 = arith.cmpi ugt, %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t010_cmp_uge_i32
module @t010_cmp_uge_i32 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.gte.i32.u %[[ARG0]], %[[ARG1]] : i32
    %1 = arith.cmpi uge, %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t011_cmp_uge_i64
module @t011_cmp_uge_i64 {

module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0: i64, %arg1 : i64) -> (i1) {
    // CHECK: vm.cmp.gte.i64.u %[[ARG0]], %[[ARG1]] : i64
    %1 = arith.cmpi uge, %arg0, %arg1 : i64
    return %1 : i1
  }
}

}

// RUN: iree-opt -split-input-file -pass-pipeline='iree-convert-std-to-vm' %s | IreeFileCheck %s

// -----
// CHECK-LABEL: @t001_cmp_eq_i32
module @t001_cmp_eq_i32 {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.eq.i32 [[ARG0]], [[ARG1]] : i32
    %1 = cmpi "eq", %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t002_cmp_ne_i32
module @t002_cmp_ne_i32 {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.ne.i32 [[ARG0]], [[ARG1]] : i32
    %1 = cmpi "ne", %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t003_cmp_slt_i32
module @t003_cmp_slt_i32 {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.lt.i32.s [[ARG0]], [[ARG1]] : i32
    %1 = cmpi "slt", %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t004_cmp_sle_i32
module @t004_cmp_sle_i32 {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.lte.i32.s [[ARG0]], [[ARG1]] : i32
    %1 = cmpi "sle", %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t005_cmp_sgt_i32
module @t005_cmp_sgt_i32 {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.gt.i32.s [[ARG0]], [[ARG1]] : i32
    %1 = cmpi "sgt", %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t006_cmp_sge_i32
module @t006_cmp_sge_i32 {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.gte.i32.s [[ARG0]], [[ARG1]] : i32
    %1 = cmpi "sge", %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t007_cmp_ult_i32
module @t007_cmp_ult_i32 {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.lt.i32.u [[ARG0]], [[ARG1]] : i32
    %1 = cmpi "ult", %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t008_cmp_ule_i32
module @t008_cmp_ule_i32 {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.lte.i32.u [[ARG0]], [[ARG1]] : i32
    %1 = cmpi "ule", %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t009_cmp_ugt_i32
module @t009_cmp_ugt_i32 {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.gt.i32.u [[ARG0]], [[ARG1]] : i32
    %1 = cmpi "ugt", %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

// -----
// CHECK-LABEL: @t010_cmp_uge_i32
module @t010_cmp_uge_i32 {

module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0: i32, %arg1 : i32) -> (i1) {
    // CHECK: vm.cmp.gte.i32.u [[ARG0]], [[ARG1]] : i32
    %1 = cmpi "uge", %arg0, %arg1 : i32
    return %1 : i1
  }
}

}

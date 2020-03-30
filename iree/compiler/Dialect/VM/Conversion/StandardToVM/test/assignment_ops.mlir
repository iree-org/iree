// RUN: iree-opt -split-input-file -pass-pipeline='iree-convert-std-to-vm' %s | IreeFileCheck %s

// -----
// CHECK-LABEL: @t001_cmp_select
module @t001_cmp_select {

module @my_module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0 : i32, %arg1 : i32) -> (i32) {
    // Note that in std, cmp returns an i1 and this relies on the dialect
    // conversion framework promoting that to i32.
    // CHECK: [[CMP:%[a-zA-Z0-9]+]] = vm.cmp.eq.i32
    %1 = cmpi "eq", %arg0, %arg1 : i32
    // CHECK: vm.select.i32 [[CMP]], [[ARG0]], [[ARG1]] : i32
    %2 = select %1, %arg0, %arg1 : i32
    return %2 : i32
  }
}

}

// -----
// CHECK-LABEL: @t002_cmp_select_index
module @t002_cmp_select_index {

module @my_module {
  // CHECK: func @my_fn
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
  func @my_fn(%arg0 : index, %arg1 : index) -> (index) {
    // Note that in std, cmp returns an i1 and this relies on the dialect
    // conversion framework promoting that to i32.
    // CHECK: [[CMP:%[a-zA-Z0-9]+]] = vm.cmp.eq.i32
    %1 = cmpi "eq", %arg0, %arg1 : index
    // CHECK: vm.select.i32 [[CMP]], [[ARG0]], [[ARG1]] : i32
    %2 = select %1, %arg0, %arg1 : index
    return %2 : index
  }
}

}

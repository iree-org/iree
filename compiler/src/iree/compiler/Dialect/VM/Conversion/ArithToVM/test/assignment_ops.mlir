// RUN: iree-opt --split-input-file --iree-vm-conversion --cse --iree-vm-target-index-bits=64 --iree-vm-target-index-bits=32 %s | FileCheck %s

// -----
// CHECK-LABEL: @t001_cmp_select
module @t001_cmp_select {

module @my_module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG2:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG3:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i32) -> (i32) {
    // Note that in std, cmp returns an i1 and this relies on the dialect
    // conversion framework promoting that to i32.
    // CHECK: %[[CMP:[a-zA-Z0-9$._-]+]] = vm.cmp.eq.i32
    %1 = arith.cmpi eq, %arg0, %arg1 : i32
    // CHECK: vm.select.i32 %[[CMP]], %[[ARG2]], %[[ARG3]] : i32
    %2 = arith.select %1, %arg2, %arg3 : i32
    return %2 : i32
  }
}

}

// -----
// CHECK-LABEL: @t002_cmp_select_index
module @t002_cmp_select_index {

module @my_module {
  // CHECK: vm.func private @my_fn
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG2:[a-zA-Z0-9$._-]+]]
  // CHECK-SAME: %[[ARG3:[a-zA-Z0-9$._-]+]]
  func.func @my_fn(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index) -> (index) {
    // Note that in std, cmp returns an i1 and this relies on the dialect
    // conversion framework promoting that to i32.
    // CHECK: %[[CMP:[a-zA-Z0-9$._-]+]] = vm.cmp.eq.i32
    %1 = arith.cmpi eq, %arg0, %arg1 : index
    // CHECK: vm.select.i32 %[[CMP]], %[[ARG2]], %[[ARG3]] : i32
    %2 = arith.select %1, %arg2, %arg3 : index
    return %2 : index
  }
}

}

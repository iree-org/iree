// RUN: iree-opt -split-input-file -pass-pipeline='iree-convert-std-to-vm' %s | IreeFileCheck %s

// -----
// CHECK-LABEL: @t001_const.i32.nonzero
module @t001_const.i32.nonzero {

module {
  func @non_zero() -> (i32) {
    // CHECK: vm.const.i32 1 : i32
    %1 = constant 1 : i32
    return %1 : i32
  }
}

}

// -----
// CHECK-LABEL: @t001_const.i32.zero
module @t001_const.i32.zero {

module {
  func @zero() -> (i32) {
    // CHECK: vm.const.i32.zero : i32
    %1 = constant 0 : i32
    return %1 : i32
  }
}

}

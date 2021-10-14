// RUN: iree-opt -split-input-file -pass-pipeline='test-iree-convert-std-to-vm' %s | IreeFileCheck %s

// -----
// CHECK-LABEL: @t001_const.i32.nonzero
module @t001_const.i32.nonzero {

module {
  func @non_zero() -> (i32) {
    // CHECK: vm.const.i32 1 : i32
    %1 = arith.constant 1 : i32
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
    %1 = arith.constant 0 : i32
    return %1 : i32
  }
}

}

// -----
// CHECK-LABEL: @t002_const.f32.nonzero
module @t002_const.f32.nonzero {

module {
  func @non_zero() -> (f32) {
    // CHECK: vm.const.f32 1.000000e+00 : f32
    %1 = arith.constant 1. : f32
    return %1 : f32
  }
}

}

// -----
// CHECK-LABEL: @t003_const.f32.zero
module @t003_const.f32.zero {

module {
  func @zero() -> (f32) {
    // CHECK: vm.const.f32.zero : f32
    %1 = arith.constant 0. : f32
    return %1 : f32
  }
}

}

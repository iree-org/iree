// RUN: iree-opt --split-input-file --iree-vm-conversion --cse --iree-vm-target-index-bits=64 %s | FileCheck %s

// -----
// CHECK-LABEL: @t001_const.i32.nonzero
module @t001_const.i32.nonzero {

module {
  func.func @non_zero() -> (i32) {
    // CHECK: vm.const.i32 1
    %1 = arith.constant 1 : i32
    return %1 : i32
  }
}

}

// -----
// CHECK-LABEL: @t001_const.i32.zero
module @t001_const.i32.zero {

module {
  func.func @zero() -> (i32) {
    // CHECK: vm.const.i32.zero
    %1 = arith.constant 0 : i32
    return %1 : i32
  }
}

}

// -----
// CHECK-LABEL: @t002_const.f32.nonzero
module @t002_const.f32.nonzero {

module {
  func.func @non_zero() -> (f32) {
    // CHECK: vm.const.f32 1.000000e+00
    %1 = arith.constant 1. : f32
    return %1 : f32
  }
}

}

// -----
// CHECK-LABEL: @t003_const.f32.zero
module @t003_const.f32.zero {

module {
  func.func @zero() -> (f32) {
    // CHECK: vm.const.f32.zero
    %1 = arith.constant 0. : f32
    return %1 : f32
  }
}

}

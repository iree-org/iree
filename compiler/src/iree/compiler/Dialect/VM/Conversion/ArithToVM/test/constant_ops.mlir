// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-vm-conversion{index-bits=64},cse)' %s | FileCheck %s

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

// -----
// CHECK-LABEL: @narrow_float_constants
module @narrow_float_constants {
module {
  // CHECK-LABEL: @half_negative_zero
  // CHECK: %[[BITS:.+]] = vm.const.i32 32768
  // CHECK: vm.return %[[BITS]] : i32
  func.func @half_negative_zero() -> f16 {
    %value = arith.constant -0.0 : f16
    return %value : f16
  }
  // CHECK-LABEL: @bfloat_three
  // CHECK: %[[BITS:.+]] = vm.const.i32 16448
  // CHECK: vm.return %[[BITS]] : i32
  func.func @bfloat_three() -> bf16 {
    %value = arith.constant 3.0 : bf16
    return %value : bf16
  }
  // CHECK-LABEL: @half_nan
  // CHECK: %[[BITS:.+]] = vm.const.i32 32256
  // CHECK: vm.return %[[BITS]] : i32
  func.func @half_nan() -> f16 {
    %value = arith.constant 0x7E00 : f16
    return %value : f16
  }
}
}

// -----
// CHECK-LABEL: @negative_zero_constants
module @negative_zero_constants {
module {
  // CHECK-LABEL: @negative_zero_f32
  // CHECK: vm.const.f32 -0.000000e+00
  func.func @negative_zero_f32() -> f32 {
    %value = arith.constant -0.0 : f32
    return %value : f32
  }
  // CHECK-LABEL: @negative_zero_f64
  // CHECK: vm.const.f64 -0.000000e+00
  func.func @negative_zero_f64() -> f64 {
    %value = arith.constant -0.0 : f64
    return %value : f64
  }
}
}

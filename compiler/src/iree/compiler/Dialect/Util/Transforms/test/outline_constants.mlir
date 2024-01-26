// RUN: iree-opt --split-input-file --iree-util-outline-constants %s | FileCheck %s

// CHECK-LABEL: @scalarConstant
func.func @scalarConstant() {
  // CHECK: = arith.constant 0 : i32
  %cst = arith.constant 0 : i32
  return
}

// -----

// CHECK-LABEL: @splatConstant
func.func @splatConstant() {
  // CHECK: = arith.constant dense<1.200000e+00> : tensor<512x128xf32>
  %cst = arith.constant dense<1.2> : tensor<512x128xf32>
  return
}

// -----

//       CHECK: util.global private @_constant {inlining_policy = #util.inline.never} = dense<[0.0287729427, 0.0297581609]> : tensor<2xf32>
//  CHECK-NEXT: util.global private @_constant_0 {inlining_policy = #util.inline.never} = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]> : tensor<8xf32>
// CHECK-LABEL: @denseConstants
func.func @denseConstants() {
  // CHECK: = util.global.load @_constant : tensor<2xf32>
  %cst_0 = arith.constant dense<[0.0287729427, 0.0297581609]> : tensor<2xf32>
  // CHECK-NEXT: = util.global.load @_constant_0 : tensor<8xf32>
  %cst_1 = arith.constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]> : tensor<8xf32>
  return
}

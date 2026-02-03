// RUN: iree-opt --split-input-file \
// RUN: --pass-pipeline='builtin.module(symbol-dce)' \
// RUN: %s | FileCheck %s
// CHECK-LABEL: @first_case
util.func @first_case(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[C0:.+]] = arith.constant 0
  %c0 = arith.constant 0 : index
  // CHECK: util.return %arg0
  util.return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @second_case
util.func @second_case(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: %[[C1:.+]] = arith.constant 1
  %c1 = arith.constant 1 : index
  // CHECK: %[[C2:.+]] = arith.constant 2
  %c2 = arith.constant 2 : index
  // CHECK: util.return %arg0
  util.return %arg0 : tensor<8xf32>
}

// -----
// CHECK-LABEL: @third_case
util.func @third_case(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  // CHECK: %[[RESULT:.+]] = arith.addf
  %result = arith.addf %arg0, %arg0 : tensor<16xf32>
  // CHECK: util.return %[[RESULT]]
  util.return %result : tensor<16xf32>
}

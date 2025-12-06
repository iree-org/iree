// RUN: iree-opt --split-input-file %s | FileCheck %s

// Test case following best practices (should have no issues).
// CHECK-LABEL: util.func @good_example
util.func @good_example(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[RESULT:.+]] = arith.addf %[[ARG:.+]], %[[ARG]]
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  // CHECK: util.return %[[RESULT]]
  util.return %0 : tensor<4xf32>
}

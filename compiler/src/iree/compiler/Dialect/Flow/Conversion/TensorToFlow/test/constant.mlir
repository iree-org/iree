// RUN: iree-opt --allow-unregistered-dialect --split-input-file --iree-flow-convert-to-flow %s | FileCheck %s

util.func public @non_tensor_constant() -> i32 {
  // CHECK: arith.constant 4
  %0 = arith.constant 4 : i32
  util.return %0 : i32
}

// -----

util.func public @tensor_constant() -> tensor<4xi32> {
  // CHECK: %[[RESULT:.+]] = flow.tensor.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  // CHECK: util.return %[[RESULT]]
  %0 = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  util.return %0 : tensor<4xi32>
}

// RUN: iree-opt %s | FileCheck %s

// CHECK-LABEL: @simple_function
util.func @simple_function(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[C0:.+]] = arith.constant 0
  %c0 = arith.constant 0 : index
  // CHECK: util.return %arg0
  util.return %arg0 : tensor<4xf32>
}

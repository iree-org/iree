// RUN: iree-opt --iree-auto-input-conversion %s | FileCheck %s

// Check that the input conversion pipeline handles a simple input and does not crash.

// CHECK-LABEL: func.func @simple_add
// CHECK:  arith.addi
func.func @simple_add(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

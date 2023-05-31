// RUN: iree-opt --iree-auto-input-conversion --split-input-file %s | FileCheck %s

// Check that the input conversion pipeline handles a simple input and does not crash.

// CHECK-LABEL: func.func @simple_add_stablehlo
// CHECK:  arith.addi
func.func @simple_add_stablehlo(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = stablehlo.add %arg0, %arg1 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func.func @simple_add_mhlo
// CHECK:  arith.addi
func.func @simple_add_mhlo(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

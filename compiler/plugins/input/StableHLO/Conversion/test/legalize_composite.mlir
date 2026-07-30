// RUN: iree-opt --split-input-file --iree-stablehlo-input-transformation-pipeline %s | FileCheck %s

// Composite ops must be replaced by their decomposition, which is then inlined
// and lowered like any other StableHLO computation.

// CHECK-LABEL: func.func @acos_composite
// CHECK-NOT:     stablehlo.composite
// CHECK-NOT:     func.call
// CHECK:         linalg.generic
// CHECK:           math.atan2
func.func @acos_composite(%arg0: tensor<8x32xf32>) -> tensor<8x32xf32> {
  %0 = stablehlo.composite "chlo.acos" %arg0 {decomposition = @chlo.acos.impl, version = 1 : i32} : (tensor<8x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}
func.func private @chlo.acos.impl(%arg0: tensor<8x32xf32>) -> tensor<8x32xf32> {
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<8x32xf32>
  %0 = stablehlo.subtract %cst, %arg0 : tensor<8x32xf32>
  %1 = stablehlo.add %cst, %arg0 : tensor<8x32xf32>
  %2 = stablehlo.multiply %0, %1 : tensor<8x32xf32>
  %3 = stablehlo.sqrt %2 : tensor<8x32xf32>
  %4 = stablehlo.atan2 %3, %arg0 : tensor<8x32xf32>
  return %4 : tensor<8x32xf32>
}

// -----

// A composite taking multiple operands and returning multiple results.

// CHECK-LABEL: func.func @multi_result_composite
// CHECK-NOT:     stablehlo.composite
// CHECK-NOT:     func.call
// CHECK:         arith.addf
// CHECK:         arith.mulf
func.func @multi_result_composite(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>)
    -> (tensor<4xf32>, tensor<4xf32>) {
  %0:2 = stablehlo.composite "my_namespace.add_mul" %arg0, %arg1 {
    decomposition = @add_mul.impl,
    composite_attributes = {my_attribute = "my_value"},
    version = 1 : i32
  } : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
}
func.func private @add_mul.impl(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>)
    -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
  %1 = stablehlo.multiply %arg0, %arg1 : tensor<4xf32>
  return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

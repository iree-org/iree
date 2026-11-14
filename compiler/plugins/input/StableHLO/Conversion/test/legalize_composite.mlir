// RUN: iree-opt --split-input-file --iree-stablehlo-input-transformation-pipeline %s | FileCheck %s --implicit-check-not=stablehlo.composite --implicit-check-not=func.call

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

// -----

// Omitting version selects the default composite version zero.
// CHECK-LABEL: func.func @default_version_composite
// CHECK: %[[RESULT:.+]] = linalg.generic
// CHECK: arith.addf
// CHECK: return %[[RESULT]] : tensor<4xf32>
func.func @default_version_composite(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.composite "my_namespace.add_one" %arg0 {
    decomposition = @add_one.impl
  } : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func private @add_one.impl(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %one = stablehlo.constant dense<1.0> : tensor<4xf32>
  %0 = stablehlo.add %arg0, %one : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// An unrecognized explicit version falls back to its attached decomposition.
// CHECK-LABEL: func.func @unknown_version_composite
// CHECK: %[[RESULT:.+]] = linalg.generic
// CHECK: arith.mulf
// CHECK: return %[[RESULT]] : tensor<4xf32>
func.func @unknown_version_composite(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.composite "my_namespace.scale" %arg0 {
    decomposition = @scale.impl,
    version = 99 : i32
  } : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func private @scale.impl(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %two = stablehlo.constant dense<2.0> : tensor<4xf32>
  %0 = stablehlo.multiply %arg0, %two : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// Composite decompositions may themselves contain composites.
// CHECK-LABEL: func.func @nested_composite
// CHECK: %[[INNER:.+]] = linalg.generic
// CHECK: arith.addf
// CHECK: %[[OUTER:.+]] = linalg.generic
// CHECK-SAME: ins(%[[INNER]] : tensor<4xf32>)
// CHECK: arith.mulf
// CHECK: return %[[OUTER]] : tensor<4xf32>
func.func @nested_composite(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.composite "my_namespace.outer" %arg0 {
    decomposition = @outer.impl,
    version = 1 : i32
  } : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
func.func private @outer.impl(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.composite "my_namespace.inner" %arg0 {
    decomposition = @inner.impl,
    version = 1 : i32
  } : (tensor<4xf32>) -> tensor<4xf32>
  %two = stablehlo.constant dense<2.0> : tensor<4xf32>
  %1 = stablehlo.multiply %0, %two : tensor<4xf32>
  return %1 : tensor<4xf32>
}
func.func private @inner.impl(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %one = stablehlo.constant dense<1.0> : tensor<4xf32>
  %0 = stablehlo.add %arg0, %one : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// RUN: iree-opt --split-input-file --iree-mhlo-to-linalg-on-tensors --canonicalize -cse %s | FileCheck %s

func.func @concatenate(%arg0: tensor<2x2xi32>, %arg1: tensor<2x4xi32>) -> tensor<2x9xi32> {
  %cst = mhlo.constant dense<514> : tensor<2x3xi32>
  %0 = "mhlo.concatenate"(%arg0, %cst, %arg1) {dimension = 1} : (tensor<2x2xi32>, tensor<2x3xi32>, tensor<2x4xi32>) -> tensor<2x9xi32>
  return %0 : tensor<2x9xi32>
}
// CHECK:       func.func @concatenate
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9$._-]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9$._-]+]]
// CHECK:         %[[CST:.+]] = arith.constant dense<514> : tensor<2x3xi32>
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<2x9xi32>
// CHECK:         %[[T0:.+]] = tensor.insert_slice %[[ARG0]] into %[[INIT]][0, 0] [2, 2] [1, 1]
// CHECK:         %[[T1:.+]] = tensor.insert_slice %[[CST]] into %[[T0]][0, 2] [2, 3] [1, 1]
// CHECK:         %[[T2:.+]] = tensor.insert_slice %[[ARG1]] into %[[T1]][0, 5] [2, 4] [1, 1]
// CHECK:         return %[[T2]]

// -----

// CHECK: ml_program.global private mutable @variable(dense<0> : tensor<2xi32>) : tensor<2xi32>
ml_program.global private mutable @variable(dense<0> : tensor<2xui32>) : tensor<2xui32>
// CHECK: func.func @global_types() -> tensor<2xi32>
func.func @global_types() -> tensor<2xui32> {
  // CHECK-NEXT: %[[VALUE:.+]] = ml_program.global_load @variable : tensor<2xi32>
  %0 = ml_program.global_load @variable : tensor<2xui32>
  // CHECK: return %[[VALUE]] : tensor<2xi32>
  return %0 : tensor<2xui32>
}

// -----

// CHECK: func.func @optimization_barrier
// CHECK: %[[RESULT1:.+]] = util.optimization_barrier %arg0 : tensor<3x4xf32
// CHECK: %[[RESULT2:.+]] = util.optimization_barrier %arg1 : tensor<4xi32>
// CHECK: return %[[RESULT1]], %[[RESULT2]]
func.func @optimization_barrier(%arg0: tensor<3x4xf32>, %arg1: tensor<4xi32>) -> (tensor<3x4xf32>, tensor<4xi32>) {
  %0, %1 = "mhlo.optimization_barrier"(%arg0, %arg1) : (tensor<3x4xf32>, tensor<4xi32>) -> (tensor<3x4xf32>, tensor<4xi32>)
  return %0, %1 : tensor<3x4xf32>, tensor<4xi32>
}

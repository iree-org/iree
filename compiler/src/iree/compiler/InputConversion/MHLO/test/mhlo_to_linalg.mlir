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
// CHECK: func.func @global_types() -> tensor<2xui32>
func.func @global_types() -> tensor<2xui32> {
  // CHECK-NEXT: %[[VALUE:.+]] = ml_program.global_load @variable : tensor<2xi32>
  %0 = ml_program.global_load @variable : tensor<2xui32>
  // CHECK-NEXT: %[[CAST:.*]] = tensor.bitcast %[[VALUE]] : tensor<2xi32> to tensor<2xui32>
  // CHECK: return %[[CAST]] : tensor<2xui32>
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

// -----

// CHECK: @unsigned_integer_input_output(%[[ARG0:.*]]: tensor<2x2xui32>, %[[ARG1:.*]]: tensor<2x2xui32>) -> tensor<2x2xui32>
func.func @unsigned_integer_input_output(%arg0: tensor<2x2xui32>, %arg1: tensor<2x2xui32>) -> tensor<2x2xui32> {
  // CHECK: %[[CAST0:.*]] = tensor.bitcast %[[ARG0]] : tensor<2x2xui32> to tensor<2x2xi32>
  // CHECK: %[[CAST1:.*]] = tensor.bitcast %[[ARG1]] : tensor<2x2xui32> to tensor<2x2xi32>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<2x2xi32> 
  // CHECK: %[[LINALG:.*]] = linalg.generic
  //  CHECK-SAME:       ins(%[[CAST0]], %[[CAST1]] : tensor<2x2xi32>, tensor<2x2xi32>
  //  CHECK-SAME:       outs(%[[INIT]] : tensor<2x2xi32>)
  // CHECK: ^bb0(%[[IN0:.*]]: i32, %[[IN1:.*]]: i32, %out: i32): 
  // CHECK: %[[ADD:.*]] = arith.addi %[[IN0]], %[[IN1]] : i32
  // CHECK: linalg.yield %[[ADD:.*]] : i32 
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<2x2xui32>, tensor<2x2xui32>) -> tensor<2x2xui32>
  // CHECK: %[[CAST_RESULT:.*]] = tensor.bitcast %[[LINALG]] : tensor<2x2xi32> to tensor<2x2xui32>
  // CHECK: return %[[CAST_RESULT]]
  return %0 : tensor<2x2xui32>
}

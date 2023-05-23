// RUN: iree-opt --split-input-file --iree-stablehlo-to-iree-input %s | FileCheck %s

// CHECK:       func.func @concatenate
// CHECK-SAME:    %[[ARG0:[^:]+]]
// CHECK-SAME:    %[[ARG1:[^:]+]]
func.func @concatenate(%arg0: tensor<2x2xi32>, %arg1: tensor<2x4xi32>) -> tensor<2x9xi32> {
  %cst = stablehlo.constant dense<514> : tensor<2x3xi32>
  %0 = "stablehlo.concatenate"(%arg0, %cst, %arg1) {dimension = 1} : (tensor<2x2xi32>, tensor<2x3xi32>, tensor<2x4xi32>) -> tensor<2x9xi32>
  return %0 : tensor<2x9xi32>
}
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C5:.+]] = arith.constant 5 : index
// CHECK-DAG:     %[[CST:.+]] = arith.constant dense<514> : tensor<2x3xi32>
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<2x9xi32>
// CHECK:         %[[T0:.+]] = tensor.insert_slice %[[ARG0]] into %[[INIT]][0, 0] [2, 2] [1, 1]
// CHECK:         %[[T1:.+]] = tensor.insert_slice %[[CST]] into %[[T0]][0, %[[C2]]] [2, 3] [1, 1]
// CHECK:         %[[T2:.+]] = tensor.insert_slice %[[ARG1]] into %[[T1]][0, %[[C5]]] [2, 4] [1, 1]
// CHECK:         return %[[T2]]

// -----

// CHECK: ml_program.global private mutable @variable(dense<0> : tensor<2xi32>) : tensor<2xi32>
ml_program.global private mutable @variable(dense<0> : tensor<2xui32>) : tensor<2xui32>
// CHECK: func.func @global_types() -> (tensor<2xi32> {iree.abi.encoding = tensor<2xui32>}
func.func @global_types() -> tensor<2xui32> {
  // CHECK-NEXT: %[[VALUE:.+]] = ml_program.global_load @variable : tensor<2xi32>
  %0 = ml_program.global_load @variable : tensor<2xui32>
  // CHECK: return %[[VALUE]] : tensor<2xi32>
  return %0 : tensor<2xui32>
}

// -----

// CHECK: func.func @optimization_barrier
// CHECK-SAME:    %[[ARG0:[^:]+]]
// CHECK-SAME:    %[[ARG1:[^:]+]]
func.func @optimization_barrier(%arg0: tensor<3x4xf32>, %arg1: tensor<4xi32>) -> (tensor<3x4xf32>, tensor<4xi32>) {
  %0, %1 = "stablehlo.optimization_barrier"(%arg0, %arg1) : (tensor<3x4xf32>, tensor<4xi32>) -> (tensor<3x4xf32>, tensor<4xi32>)
  return %0, %1 : tensor<3x4xf32>, tensor<4xi32>
}
// CHECK: %[[RESULT1:.+]] = util.optimization_barrier %[[ARG0]] : tensor<3x4xf32
// CHECK: %[[RESULT2:.+]] = util.optimization_barrier %[[ARG1]] : tensor<4xi32>
// CHECK: return %[[RESULT1]], %[[RESULT2]]

// -----

// CHECK: @unsigned_integer_input_output(%[[ARG0:.*]]: tensor<2x2xi32> {iree.abi.encoding = tensor<2x2xui32>}, %[[ARG1:.*]]: tensor<2x2xi32> {iree.abi.encoding = tensor<2x2xui32>}) -> (tensor<2x2xi32> {iree.abi.encoding = tensor<2x2xui32>})
func.func @unsigned_integer_input_output(%arg0: tensor<2x2xui32>, %arg1: tensor<2x2xui32>) -> tensor<2x2xui32> {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<2x2xi32>
  // CHECK: %[[RESULT:.*]] = linalg.generic
  //  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] : tensor<2x2xi32>, tensor<2x2xi32>
  //  CHECK-SAME:       outs(%[[INIT]] : tensor<2x2xi32>)
  // CHECK: ^bb0(%[[IN0:.*]]: i32, %[[IN1:.*]]: i32, %out: i32):
  // CHECK: %[[ADD:.*]] = arith.addi %[[IN0]], %[[IN1]] : i32
  // CHECK: linalg.yield %[[ADD:.*]] : i32
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<2x2xui32>, tensor<2x2xui32>) -> tensor<2x2xui32>
  // CHECK: return %[[RESULT]] : tensor<2x2xi32>
  return %0 : tensor<2x2xui32>
}

// -----

// CHECK: func.func @aliasing_output
// CHECK-SAME:    %[[ARG0:[^:]+]]: tensor<3x4xf32> {iree.abi.output = 1 : index}
// CHECK-SAME:    %[[ARG1:[^:]+]]: tensor<4xi32> {iree.abi.encoding = tensor<4xui32>}
func.func @aliasing_output(%arg0: tensor<3x4xf32> {tf.aliasing_output = 1 : i32}, %arg1: tensor<4xui32>) -> (tensor<4xui32>, tensor<3x4xf32>) {
  return %arg1, %arg0 : tensor<4xui32>, tensor<3x4xf32>
}

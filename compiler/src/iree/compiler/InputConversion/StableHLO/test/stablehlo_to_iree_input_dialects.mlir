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

// -----

// Tests that frontend attributes are stripped from the module and functions.

// CHECK: module @jax_module
module @jax_module attributes {
  // CHECK-NOT: mhlo.num_partitions
  mhlo.num_partitions = 1 : i32,
  // CHECK-NOT: mhlo.num_replicas
  mhlo.num_replicas = 1 : i32
} {
  // CHECK: func.func public @main
  func.func public @main(
      // CHECK-NOT: jax.arg_info
      // CHECK-NOT: mhlo.sharding
      %arg0: tensor<5x6xcomplex<f32>> {jax.arg_info = "array", mhlo.sharding = "{replicated}"})
      // CHECK-NOT: jax.result_info
      -> (tensor<5x6xcomplex<f32>> {jax.result_info = ""}) {
    return %arg0 : tensor<5x6xcomplex<f32>>
  }
}

// -----

// CHECK-LABEL: @empty_zero_extent
func.func public @empty_zero_extent(%arg0: tensor<ui8>, %arg1: tensor<0x4xui32>) -> (tensor<0x4xui32>) {
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<0x4xi32>
  %0 = tensor.empty() : tensor<0x4xui32>
  // CHECK: return %[[EMPTY]]
  return %0 : tensor<0x4xui32>
}

// -----

// CHECK-LABEL: @convert_return
func.func @convert_return() -> tensor<i32> {
  // CHECK: %[[CST:.+]] = arith.constant dense<1>
  %cst = arith.constant dense<1> : tensor<i32>
  // CHECK: return %[[CST]]
  stablehlo.return %cst : tensor<i32>
}

// -----

// CHECK-LABEL: @while_unsigned
func.func @while_unsigned(%arg0: tensor<ui32>) -> tensor<ui32> {
  // CHECK: scf.while
  %0 = scf.while (%arg1 = %arg0) : (tensor<ui32>) -> tensor<ui32> {
    // CHECK: linalg.generic
    %1 = stablehlo.compare  LT, %arg1, %arg1 : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
    // CHECK: tensor.extract
    %extracted = tensor.extract %1[] : tensor<i1>
    // CHECK: scf.condition
    scf.condition(%extracted) %arg1 : tensor<ui32>
  } do {
  ^bb0(%arg1: tensor<ui32>):
    // CHECK: linalg.generic
    %1 = stablehlo.add %arg1, %arg1 : tensor<ui32>
    // CHECK: scf.yield
    scf.yield %1 : tensor<ui32>
  }
  return %0 : tensor<ui32>
}

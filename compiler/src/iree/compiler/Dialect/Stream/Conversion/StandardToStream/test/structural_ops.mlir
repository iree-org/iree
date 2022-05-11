// RUN: iree-opt --split-input-file --iree-stream-conversion %s | FileCheck %s

// CHECK-LABEL: @functionExpansion
//  CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[ARG0_SIZE:.+]]: index,
//   CHECK-SAME: %[[ARG1:.+]]: i1,
//  CHECK-SAME:  %[[ARG2:.+]]: !stream.resource<*>, %[[ARG2_SIZE:.+]]: index)
//  CHECK-SAME: -> (!stream.resource<*>, index, i1, !stream.resource<*>, index)
func.func @functionExpansion(%arg0: tensor<4x?xf32>, %arg1: i1, %arg2: tensor<i32>)
    -> (tensor<4x?xf32>, i1, tensor<i32>) {
  // CHECK-NEXT: %[[RET:.+]]:5 = call @callee(%[[ARG0]], %[[ARG0_SIZE]], %[[ARG1]], %[[ARG2]], %[[ARG2_SIZE]])
  // CHECK-SAME: : (!stream.resource<*>, index, i1, !stream.resource<*>, index) -> (!stream.resource<*>, index, i1, !stream.resource<*>, index)
  %0:3 = call @callee(%arg0, %arg1, %arg2) : (tensor<4x?xf32>, i1, tensor<i32>) -> (tensor<4x?xf32>, i1, tensor<i32>)
  // CHECK: return %[[RET]]#0, %[[RET]]#1, %[[RET]]#2,  %[[RET]]#3, %[[RET]]#4 : !stream.resource<*>, index, i1, !stream.resource<*>, index
  return %0#0, %0#1, %0#2 : tensor<4x?xf32>, i1, tensor<i32>
}

// CHECK: func.func private @callee
func.func private @callee(%arg0: tensor<4x?xf32>, %arg1: i1, %arg2: tensor<i32>)
    -> (tensor<4x?xf32>, i1, tensor<i32>)

// -----

// CHECK-LABEL: @brExpansion
//  CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[ARG0_SIZE:.+]]: index, %arg2: i1)
//  CHECK-SAME: -> (!stream.resource<*>, index, i1)
func.func @brExpansion(%arg0: tensor<1xf32>, %arg1: i1) -> (tensor<1xf32>, i1) {
  // CHECK: cf.br ^bb1(%[[ARG0]], %[[ARG0_SIZE]], %arg2 : !stream.resource<*>, index, i1)
  cf.br ^bb1(%arg0, %arg1 : tensor<1xf32>, i1)
// CHECK: ^bb1(%[[BB_ARG0:.+]]: !stream.resource<*>, %[[BB_ARG1:.+]]: index, %[[BB_ARG2:.+]]: i1):
^bb1(%0: tensor<1xf32>, %1: i1):
  // CHECK: return %[[BB_ARG0]], %[[BB_ARG1]], %[[BB_ARG2]] : !stream.resource<*>, index, i1
  return %0, %1 : tensor<1xf32>, i1
}

// -----

// CHECK-LABEL: @condBrExpansion
//  CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[ARG0_SIZE:.+]]: index,
//  CHECK-SAME:  %[[ARG1:.+]]: !stream.resource<*>, %[[ARG1_SIZE:.+]]: index)
//  CHECK-SAME: -> (!stream.resource<*>, index)
func.func @condBrExpansion(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %true = arith.constant 1 : i1
  //      CHECK: cf.cond_br %true,
  // CHECK-SAME:   ^bb1(%[[ARG0]], %[[ARG0_SIZE]] : !stream.resource<*>, index),
  // CHECK-SAME:   ^bb1(%[[ARG1]], %[[ARG1_SIZE]] : !stream.resource<*>, index)
  cf.cond_br %true, ^bb1(%arg0 : tensor<1xf32>), ^bb1(%arg1 : tensor<1xf32>)
^bb1(%0: tensor<1xf32>):
  return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: @selectExpansion
//  CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[ARG0_SIZE:.+]]: index,
//  CHECK-SAME:  %[[COND:.+]]: i1,
//  CHECK-SAME:  %[[ARG1:.+]]: !stream.resource<*>, %[[ARG1_SIZE:.+]]: index)
//  CHECK-SAME: -> (!stream.resource<*>, index)
func.func @selectExpansion(%arg0: tensor<1xf32>, %cond: i1, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK-DAG: %[[RET:.+]] = arith.select %[[COND]], %[[ARG0]], %[[ARG1]] : !stream.resource<*>
  // CHECK-DAG: %[[RET_SIZE:.+]] = arith.select %[[COND]], %[[ARG0_SIZE]], %[[ARG1_SIZE]] : index
  %0 = arith.select %cond, %arg0, %arg1 : tensor<1xf32>
  // CHECK: return %[[RET]], %[[RET_SIZE]] : !stream.resource<*>, index
  return %0 : tensor<1xf32>
}

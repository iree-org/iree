// RUN: iree-opt -split-input-file -iree-stream-conversion %s | IreeFileCheck %s

// CHECK-LABEL: @functionExpansion
//  CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[ARG0_SIZE:.+]]: index,
//   CHECK-SAME: %[[ARG1:.+]]: i1,
//  CHECK-SAME:  %[[ARG2:.+]]: !stream.resource<*>, %[[ARG2_SIZE:.+]]: index)
//  CHECK-SAME: -> (!stream.resource<*>, index, i1, !stream.resource<*>, index)
func @functionExpansion(%arg0: tensor<4x?xf32>, %arg1: i1, %arg2: tensor<i32>)
    -> (tensor<4x?xf32>, i1, tensor<i32>) {
  //  CHECK-DAG: %[[ARG0_T:.+]] = stream.async.transfer %arg0 : !stream.resource<*>{%[[ARG0_SIZE]]} -> !stream.resource<*>{%[[ARG0_SIZE]]}
  //  CHECK-DAG: %[[ARG2_T:.+]] = stream.async.transfer %arg3 : !stream.resource<*>{%[[ARG2_SIZE]]} -> !stream.resource<*>{%[[ARG2_SIZE]]}
  // CHECK-NEXT: %[[RET:.+]]:5 = call @callee(%[[ARG0_T]], %[[ARG0_SIZE]], %[[ARG1]], %[[ARG2_T]], %[[ARG2_SIZE]])
  // CHECK-SAME: : (!stream.resource<*>, index, i1, !stream.resource<*>, index) -> (!stream.resource<*>, index, i1, !stream.resource<*>, index)
  %0:3 = call @callee(%arg0, %arg1, %arg2) : (tensor<4x?xf32>, i1, tensor<i32>) -> (tensor<4x?xf32>, i1, tensor<i32>)
  //  CHECK-DAG: %[[RET0_T:.+]] = stream.async.transfer %[[RET]]#0 : !stream.resource<*>{%[[RET]]#1} -> !stream.resource<*>{%[[RET]]#1}
  //  CHECK-DAG: %[[RET3_T:.+]] = stream.async.transfer %[[RET]]#3 : !stream.resource<*>{%[[RET]]#4} -> !stream.resource<*>{%[[RET]]#4}
  // CHECK: return %[[RET0_T]], %[[RET]]#1, %[[RET]]#2, %[[RET3_T]], %[[RET]]#4 : !stream.resource<*>, index, i1, !stream.resource<*>, index
  return %0#0, %0#1, %0#2 : tensor<4x?xf32>, i1, tensor<i32>
}

// CHECK: func private @callee
func private @callee(%arg0: tensor<4x?xf32>, %arg1: i1, %arg2: tensor<i32>)
    -> (tensor<4x?xf32>, i1, tensor<i32>)

// -----

// CHECK-LABEL: @brExpansion
//  CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[ARG0_SIZE:.+]]: index, %arg2: i1)
//  CHECK-SAME: -> (!stream.resource<*>, index, i1)
func @brExpansion(%arg0: tensor<1xf32>, %arg1: i1) -> (tensor<1xf32>, i1) {
  // CHECK: %[[ARG0_T:.+]] = stream.async.transfer %[[ARG0]] : !stream.resource<*>{%[[ARG0_SIZE]]} -> !stream.resource<*>{%[[ARG0_SIZE]]}
  // CHECK: br ^bb1(%[[ARG0_T]], %[[ARG0_SIZE]], %arg2 : !stream.resource<*>, index, i1)
  br ^bb1(%arg0, %arg1 : tensor<1xf32>, i1)
// CHECK: ^bb1(%[[BB_ARG0:.+]]: !stream.resource<*>, %[[BB_ARG1:.+]]: index, %[[BB_ARG2:.+]]: i1):
^bb1(%0: tensor<1xf32>, %1: i1):
  // CHECK: %[[RET0_T:.+]] = stream.async.transfer %[[BB_ARG0]] : !stream.resource<*>{%[[BB_ARG1]]} -> !stream.resource<*>{%[[BB_ARG1]]}
  // CHECK: return %[[RET0_T]], %[[BB_ARG1]], %[[BB_ARG2]] : !stream.resource<*>, index, i1
  return %0, %1 : tensor<1xf32>, i1
}

// -----

// CHECK-LABEL: @condBrExpansion
//  CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[ARG0_SIZE:.+]]: index,
//  CHECK-SAME:  %[[ARG1:.+]]: !stream.resource<*>, %[[ARG1_SIZE:.+]]: index)
//  CHECK-SAME: -> (!stream.resource<*>, index)
func @condBrExpansion(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  //      CHECK: %[[ARG0_T:.+]] = stream.async.transfer %[[ARG0]] : !stream.resource<*>{%[[ARG0_SIZE]]} -> !stream.resource<*>{%[[ARG0_SIZE]]}
  //      CHECK: %[[ARG1_T:.+]] = stream.async.transfer %[[ARG1]] : !stream.resource<*>{%[[ARG1_SIZE]]} -> !stream.resource<*>{%[[ARG1_SIZE]]}
  %true = arith.constant 1 : i1
  //      CHECK: cond_br %true,
  // CHECK-SAME:   ^bb1(%[[ARG0_T]], %[[ARG0_SIZE]] : !stream.resource<*>, index),
  // CHECK-SAME:   ^bb1(%[[ARG1_T]], %[[ARG1_SIZE]] : !stream.resource<*>, index)
  cond_br %true, ^bb1(%arg0 : tensor<1xf32>), ^bb1(%arg1 : tensor<1xf32>)
^bb1(%0: tensor<1xf32>):
  return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: @selectExpansion
//  CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[ARG0_SIZE:.+]]: index,
//  CHECK-SAME:  %[[COND:.+]]: i1,
//  CHECK-SAME:  %[[ARG1:.+]]: !stream.resource<*>, %[[ARG1_SIZE:.+]]: index)
//  CHECK-SAME: -> (!stream.resource<*>, index)
func @selectExpansion(%arg0: tensor<1xf32>, %cond: i1, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  //      CHECK: %[[ARG0_T:.+]] = stream.async.transfer %[[ARG0]] : !stream.resource<*>{%[[ARG0_SIZE]]} -> !stream.resource<*>{%[[ARG0_SIZE]]}
  //      CHECK: %[[ARG1_T:.+]] = stream.async.transfer %[[ARG1]] : !stream.resource<*>{%[[ARG1_SIZE]]} -> !stream.resource<*>{%[[ARG1_SIZE]]}
  // CHECK: %[[RET:.+]] = select %[[COND]], %[[ARG0_T]], %[[ARG1_T]] : !stream.resource<*>
  %0 = select %cond, %arg0, %arg1 : tensor<1xf32>
  // CHECK: %[[RET_SIZE:.+]] = stream.resource.size %[[RET]] : !stream.resource<*>
  // CHECK: return %[[RET]], %[[RET_SIZE]] : !stream.resource<*>, index
  return %0 : tensor<1xf32>
}

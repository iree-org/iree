// RUN: iree-opt --split-input-file --iree-stream-conversion %s | FileCheck %s

// CHECK-LABEL: @brExpansion
//  CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[ARG0_SIZE:.+]]: index, %arg2: i1)
//  CHECK-SAME: -> (!stream.resource<*>, index, i1)
util.func public @brExpansion(%arg0: tensor<1xf32>, %arg1: i1) -> (tensor<1xf32>, i1) {
  // CHECK: cf.br ^bb1(%[[ARG0]], %[[ARG0_SIZE]], %arg2 : !stream.resource<*>, index, i1)
  cf.br ^bb1(%arg0, %arg1 : tensor<1xf32>, i1)
// CHECK: ^bb1(%[[BB_ARG0:.+]]: !stream.resource<*>, %[[BB_ARG1:.+]]: index, %[[BB_ARG2:.+]]: i1):
^bb1(%0: tensor<1xf32>, %1: i1):
  // CHECK: util.return %[[BB_ARG0]], %[[BB_ARG1]], %[[BB_ARG2]] : !stream.resource<*>, index, i1
  util.return %0, %1 : tensor<1xf32>, i1
}

// -----

// CHECK-LABEL: @condBrExpansion
//  CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[ARG0_SIZE:.+]]: index,
//  CHECK-SAME:  %[[ARG1:.+]]: !stream.resource<*>, %[[ARG1_SIZE:.+]]: index)
//  CHECK-SAME: -> (!stream.resource<*>, index)
util.func public @condBrExpansion(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %true = arith.constant 1 : i1
  //      CHECK: cf.cond_br %true,
  // CHECK-SAME:   ^bb1(%[[ARG0]], %[[ARG0_SIZE]] : !stream.resource<*>, index),
  // CHECK-SAME:   ^bb1(%[[ARG1]], %[[ARG1_SIZE]] : !stream.resource<*>, index)
  cf.cond_br %true, ^bb1(%arg0 : tensor<1xf32>), ^bb1(%arg1 : tensor<1xf32>)
^bb1(%0: tensor<1xf32>):
  util.return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: @switchExpansion
//  CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[ARG0_SIZE:.+]]: index,
//  CHECK-SAME:  %[[ARG1:.+]]: !stream.resource<*>, %[[ARG1_SIZE:.+]]: index)
//  CHECK-SAME: -> (!stream.resource<*>, index)
util.func public @switchExpansion(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %flag = arith.constant 1 : i32
  //      CHECK: %[[FLAG:.+]] = arith.constant 1 : i32
  //      CHECK: cf.switch %[[FLAG]] : i32, [
  // CHECK-NEXT:   default: ^bb1(%[[ARG0]], %[[ARG0_SIZE]] : !stream.resource<*>, index),
  // CHECK-NEXT:   0: ^bb2(%[[ARG1]], %[[ARG1_SIZE]] : !stream.resource<*>, index)
  // CHECK-NEXT: ]
  cf.switch %flag : i32, [
    default: ^bb1(%arg0 : tensor<1xf32>),
    0: ^bb2(%arg1 : tensor<1xf32>)
  ]
^bb1(%0: tensor<1xf32>):
  util.return %0 : tensor<1xf32>
^bb2(%1: tensor<1xf32>):
  util.return %1 : tensor<1xf32>
}

// -----

// CHECK-LABEL: @selectExpansion
//  CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[ARG0_SIZE:.+]]: index,
//  CHECK-SAME:  %[[COND:.+]]: i1,
//  CHECK-SAME:  %[[ARG1:.+]]: !stream.resource<*>, %[[ARG1_SIZE:.+]]: index)
//  CHECK-SAME: -> (!stream.resource<*>, index)
util.func public @selectExpansion(%arg0: tensor<1xf32>, %cond: i1, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK-DAG: %[[RET:.+]] = arith.select %[[COND]], %[[ARG0]], %[[ARG1]] : !stream.resource<*>
  // CHECK-DAG: %[[RET_SIZE:.+]] = arith.select %[[COND]], %[[ARG0_SIZE]], %[[ARG1_SIZE]] : index
  %0 = arith.select %cond, %arg0, %arg1 : tensor<1xf32>
  // CHECK: util.return %[[RET]], %[[RET_SIZE]] : !stream.resource<*>, index
  util.return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: @scfIfExpansion
// CHECK-SAME: %[[COND:.+]]: i1, %[[ARG0:.+]]: !stream.resource<*>, %[[IDX0:.+]]: index, %[[ARG1:.+]]: !stream.resource<*>, %[[IDX1:.+]]: index
util.func public @scfIfExpansion(%cond: i1, %arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK: %[[IF:.+]]:2 = scf.if %arg0 -> (!stream.resource<*>, index)
  %0 = scf.if %cond -> tensor<1xf32> {
    // CHECK: scf.yield %[[ARG0]], %[[IDX0]]
    scf.yield %arg0 : tensor<1xf32>
  } else {
    // CHECK: scf.yield %[[ARG1]], %[[IDX1]]
    scf.yield %arg1 : tensor<1xf32>
  }
  // CHECK: util.return %[[IF]]#0, %[[IF]]#1
  util.return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: @scfWhileExpansion
// CHECK-SAME: %[[ARG0:.+]]: i32, %[[ARG1:.+]]: !stream.resource<*>, %[[ARG2:.+]]: index
util.func public @scfWhileExpansion(%arg0 : i32, %arg1 : tensor<1xf32>) {
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i32
  // CHECK: scf.while
  // CHECK-SAME: (%[[ARG3:.+]] = %[[ARG0]], %[[ARG4:.+]] = %[[ARG1]], %[[ARG5:.+]] = %[[ARG2]])
  // CHECK-SAME: (i32, !stream.resource<*>, index) -> (i32, !stream.resource<*>, index)
  %0:2 = scf.while (%arg2 = %arg0, %arg3 = %arg1) : (i32, tensor<1xf32>) -> (i32, tensor<1xf32>) {
    %1 = arith.cmpi slt, %arg2, %c10 : i32
    // CHECK: scf.condition(%[[V:.+]]) %[[ARG3]], %[[ARG1]], %[[ARG2]] : i32, !stream.resource<*>, index
    scf.condition(%1) %arg2, %arg1 : i32, tensor<1xf32>
  } do {
  // CHECK: ^bb0(%[[ARG3:.+]]: i32, %[[ARG4:.+]]: !stream.resource<*>, %[[ARG5:.+]]: index):
  ^bb0(%arg2: i32, %arg3 : tensor<1xf32>):
    %1 = arith.addi %arg2, %c1 : i32
  // CHECK: scf.yield %[[V:.+]], %[[ARG1]], %[[ARG2]] : i32, !stream.resource<*>, index
    scf.yield %1, %arg1 : i32, tensor<1xf32>
  }
  util.return
}

// -----

// CHECK-LABEL: @scfForExpansion
// CHECK-SAME: %[[ARG0:.+]]: index,
// CHECK-SAME: %[[ARG1:.+]]: !stream.resource<*>,
// CHECK-SAME: %[[ARG2:.+]]: index
util.func public @scfForExpansion(%arg0 : index, %arg1 : tensor<1xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: %[[C1:.+]] = arith.constant 1 : index
  // CHECK: [[FOR:.+]]:2 = scf.for %[[ARG3:.+]] = %[[C0]] to %[[ARG0]] step %c1 iter_args(%[[ARG4:.+]] = %[[ARG1]], %[[ARG5:.+]] = %[[ARG2]]) -> (!stream.resource<*>, index)
  scf.for %i = %c0 to %arg0 step %c1 iter_args(%arg2 = %arg1) -> (tensor<1xf32>) {
    scf.yield %arg2 : tensor<1xf32>
  }
  util.return
}

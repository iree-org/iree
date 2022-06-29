// RUN: iree-opt --split-input-file --iree-hal-conversion %s | FileCheck %s

// These patterns are not doing anything HAL-specific and instead just allowing
// for the ops to update their types during dialect conversions. These should be
// moved to a general utility location or really become something upstream that
// can be reused.

// CHECK-LABEL: @funcOp
// CHECK-SAME: (%[[ARG0:.+]]: !hal.buffer_view) -> !hal.buffer_view
func.func @funcOp(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  // CHECK: return %[[ARG0]] : !hal.buffer_view
  return %arg0 : tensor<4x2xf32>
}

// -----

// CHECK-LABEL: @callOp
// CHECK-SAME: (%[[ARG0:.+]]: !hal.buffer_view) -> !hal.buffer_view
func.func @callOp(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  // CHECK: %[[RET0:.+]] = call @extern(%[[ARG0]]) : (!hal.buffer_view) -> !hal.buffer_view
  %ret0 = call @extern(%arg0) : (tensor<4x2xf32>) -> tensor<4x2xf32>
  // CHECK: return %[[RET0]] : !hal.buffer_view
  return %ret0 : tensor<4x2xf32>
}
// CHECK: func.func private @extern(!hal.buffer_view) -> !hal.buffer_view
func.func private @extern(tensor<4x2xf32>) -> tensor<4x2xf32>

// -----

// CHECK-LABEL: @brOp
// CHECK-SAME: (%[[ARG0:.+]]: !hal.buffer_view) -> !hal.buffer_view
func.func @brOp(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  // CHECK: cf.br ^bb1(%[[ARG0]] : !hal.buffer_view)
  cf.br ^bb1(%arg0 : tensor<4x2xf32>)
// CHECK: ^bb1(%[[BB1_ARG0:.+]]: !hal.buffer_view):
^bb1(%bb1_arg0: tensor<4x2xf32>):
  // CHECK: return %[[BB1_ARG0]] : !hal.buffer_view
  return %bb1_arg0 : tensor<4x2xf32>
}

// -----

// CHECK-LABEL: @condBrOp
// CHECK-SAME: (%[[COND:.+]]: i1, %[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: !hal.buffer_view) -> !hal.buffer_view
func.func @condBrOp(%cond: i1, %arg0: tensor<4x2xf32>, %arg1: tensor<4x2xf32>) -> tensor<4x2xf32> {
  // CHECK: cf.cond_br %[[COND]], ^bb1(%[[ARG0]] : !hal.buffer_view), ^bb1(%[[ARG1]] : !hal.buffer_view)
  cf.cond_br %cond, ^bb1(%arg0 : tensor<4x2xf32>), ^bb1(%arg1 : tensor<4x2xf32>)
// CHECK: ^bb1(%[[BB1_ARG0:.+]]: !hal.buffer_view):
^bb1(%bb1_arg0 : tensor<4x2xf32>):
  // CHECK: return %[[BB1_ARG0]] : !hal.buffer_view
  return %bb1_arg0 : tensor<4x2xf32>
}

// -----

// CHECK-LABEL: @selectOp
// CHECK-SAME: (%[[COND:.+]]: i1, %[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: !hal.buffer_view) -> !hal.buffer_view
func.func @selectOp(%cond: i1, %arg0: tensor<4x2xf32>, %arg1: tensor<4x2xf32>) -> tensor<4x2xf32> {
  // CHECK: %[[RET0:.+]] = arith.select %[[COND]], %[[ARG0]], %[[ARG1]] : !hal.buffer_view
  %ret0 = arith.select %cond, %arg0, %arg1 : tensor<4x2xf32>
  // CHECK: return %[[RET0]] : !hal.buffer_view
  return %ret0 : tensor<4x2xf32>
}

// -----

// CHECK-LABEL: @ifOp
// CHECK-SAME: (%[[COND:.+]]: i1, %[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: !hal.buffer_view) -> !hal.buffer_view
func.func @ifOp(%cond: i1, %arg0: tensor<4x2xf32>, %arg1: tensor<4x2xf32>) -> tensor<4x2xf32> {
  // CHECK: %[[RET0:.+]] = scf.if %[[COND]] -> (!hal.buffer_view)
  %ret0 = scf.if %cond -> (tensor<4x2xf32>) {
    // CHECK: scf.yield %[[ARG0]] : !hal.buffer_view
    scf.yield %arg0 : tensor<4x2xf32>
  } else {
    // CHECK: scf.yield %[[ARG1]] : !hal.buffer_view
    scf.yield %arg1 : tensor<4x2xf32>
  }
  // CHECK: return %[[RET0]] : !hal.buffer_view
  return %ret0 : tensor<4x2xf32>
}

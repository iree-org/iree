// RUN: iree-opt --split-input-file --iree-util-test-conversion %s | FileCheck %s

// These patterns are not doing anything dialect-specific and instead just
// allowing for the ops to update their types during dialect conversions.

// CHECK: util.initializer
util.initializer {
  // CHECK: %[[VALUE:.+]] = func.call @extern
  %value = func.call @extern() : () -> memref<?xi8>
  // CHECK: cf.br ^bb1(%[[VALUE]] : !util.buffer)
  cf.br ^bb1(%value : memref<?xi8>)
// CHECK: ^bb1(%[[ARG:.+]]: !util.buffer)
^bb1(%block_arg: memref<?xi8>):
  util.initializer.return
}
func.func private @extern() -> memref<?xi8>

// -----

// CHECK-LABEL: @funcOp
// CHECK-SAME: (%[[ARG0:.+]]: !util.buffer) -> !util.buffer
func.func @funcOp(%arg0: memref<?xi8>) -> memref<?xi8> {
  // CHECK: return %[[ARG0]] : !util.buffer
  return %arg0 : memref<?xi8>
}

// -----

// CHECK-LABEL: @callOp
// CHECK-SAME: (%[[ARG0:.+]]: !util.buffer) -> !util.buffer
func.func @callOp(%arg0: memref<?xi8>) -> memref<?xi8> {
  // CHECK: %[[RET0:.+]] = call @extern(%[[ARG0]]) : (!util.buffer) -> !util.buffer
  %ret0 = call @extern(%arg0) : (memref<?xi8>) -> memref<?xi8>
  // CHECK: return %[[RET0]] : !util.buffer
  return %ret0 : memref<?xi8>
}
// CHECK: func.func private @extern(!util.buffer) -> !util.buffer
func.func private @extern(memref<?xi8>) -> memref<?xi8>

// -----

// CHECK-LABEL: @brOp
// CHECK-SAME: (%[[ARG0:.+]]: !util.buffer) -> !util.buffer
func.func @brOp(%arg0: memref<?xi8>) -> memref<?xi8> {
  // CHECK: cf.br ^bb1(%[[ARG0]] : !util.buffer)
  cf.br ^bb1(%arg0 : memref<?xi8>)
// CHECK: ^bb1(%[[BB1_ARG0:.+]]: !util.buffer):
^bb1(%bb1_arg0: memref<?xi8>):
  // CHECK: return %[[BB1_ARG0]] : !util.buffer
  return %bb1_arg0 : memref<?xi8>
}

// -----

// CHECK-LABEL: @condBrOp
// CHECK-SAME: (%[[COND:.+]]: i1, %[[ARG0:.+]]: !util.buffer, %[[ARG1:.+]]: !util.buffer) -> !util.buffer
func.func @condBrOp(%cond: i1, %arg0: memref<?xi8>, %arg1: memref<?xi8>) -> memref<?xi8> {
  // CHECK: cf.cond_br %[[COND]], ^bb1(%[[ARG0]] : !util.buffer), ^bb1(%[[ARG1]] : !util.buffer)
  cf.cond_br %cond, ^bb1(%arg0 : memref<?xi8>), ^bb1(%arg1 : memref<?xi8>)
// CHECK: ^bb1(%[[BB1_ARG0:.+]]: !util.buffer):
^bb1(%bb1_arg0 : memref<?xi8>):
  // CHECK: return %[[BB1_ARG0]] : !util.buffer
  return %bb1_arg0 : memref<?xi8>
}

// -----

// CHECK-LABEL: @switchOp
// CHECK-SAME: (%[[FLAG:.+]]: i32, %[[ARG0:.+]]: !util.buffer, %[[ARG1:.+]]: !util.buffer) -> !util.buffer
func.func @switchOp(%flag: i32, %arg0: memref<?xi8>, %arg1: memref<?xi8>) -> memref<?xi8> {
  // CHECK: cf.switch %[[FLAG]] : i32, [
  // CHECK:   default: ^bb1(%[[ARG0]] : !util.buffer),
  // CHECK:   0: ^bb1(%[[ARG1]] : !util.buffer)
  // CHECK: ]
  cf.switch %flag : i32, [
    default: ^bb1(%arg0 : memref<?xi8>),
    0: ^bb1(%arg1 : memref<?xi8>)
  ]
// CHECK: ^bb1(%[[BB1_ARG0:.+]]: !util.buffer):
^bb1(%bb1_arg0 : memref<?xi8>):
  // CHECK: return %[[BB1_ARG0]] : !util.buffer
  return %bb1_arg0 : memref<?xi8>
}

// -----

// CHECK-LABEL: @selectOp
// CHECK-SAME: (%[[COND:.+]]: i1, %[[ARG0:.+]]: !util.buffer, %[[ARG1:.+]]: !util.buffer) -> !util.buffer
func.func @selectOp(%cond: i1, %arg0: memref<?xi8>, %arg1: memref<?xi8>) -> memref<?xi8> {
  // CHECK: %[[RET0:.+]] = arith.select %[[COND]], %[[ARG0]], %[[ARG1]] : !util.buffer
  %ret0 = arith.select %cond, %arg0, %arg1 : memref<?xi8>
  // CHECK: return %[[RET0]] : !util.buffer
  return %ret0 : memref<?xi8>
}

// -----

// CHECK-LABEL: @ifOp
// CHECK-SAME: (%[[COND:.+]]: i1, %[[ARG0:.+]]: !util.buffer, %[[ARG1:.+]]: !util.buffer) -> !util.buffer
func.func @ifOp(%cond: i1, %arg0: memref<?xi8>, %arg1: memref<?xi8>) -> memref<?xi8> {
  // CHECK: %[[RET0:.+]] = scf.if %[[COND]] -> (!util.buffer)
  %ret0 = scf.if %cond -> (memref<?xi8>) {
    // CHECK: scf.yield %[[ARG0]] : !util.buffer
    scf.yield %arg0 : memref<?xi8>
  } else {
    // CHECK: scf.yield %[[ARG1]] : !util.buffer
    scf.yield %arg1 : memref<?xi8>
  }
  // CHECK: return %[[RET0]] : !util.buffer
  return %ret0 : memref<?xi8>
}

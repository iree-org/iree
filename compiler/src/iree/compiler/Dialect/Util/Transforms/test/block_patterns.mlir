// RUN: iree-opt --split-input-file --iree-util-apply-patterns --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @foldBrArguments
// CHECK-SAME: (%[[COND:.+]]: i1, %[[ARG1:.+]]: index)
func.func @foldBrArguments(%cond: i1, %arg1: index) -> index {
  // CHECK: cf.cond_br %[[COND]]
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  // CHECK: %[[OP1:.+]] = "some.op1"
  %0 = "some.op1"() : () -> index
  // CHECK: cf.br ^bb3(%[[OP1]], %[[OP1]], %[[ARG1]] : index, index, index)
  cf.br ^bb3(%0, %0, %arg1, %0 : index, index, index, index)
^bb2:
  // CHECK: %[[OP2:.+]] = "some.op2"
  %1 = "some.op2"() : () -> index
  // CHECK: cf.br ^bb3(%[[ARG1]], %[[OP2]], %[[OP2]] : index, index, index)
  cf.br ^bb3(%arg1, %1, %1, %1 : index, index, index, index)
// CHECK: ^bb3(%[[BB3_ARG0:.+]]: index, %[[BB3_ARG1:.+]]: index, %[[BB3_ARG2:.+]]: index):
^bb3(%bb3_0: index, %bb3_1: index, %bb3_2: index, %bb3_3: index):
  // CHECK: %[[OP3:.+]] = "some.op3"(%[[BB3_ARG0]], %[[BB3_ARG1]], %[[BB3_ARG2]], %[[BB3_ARG1]])
  %2 = "some.op3"(%bb3_0, %bb3_1, %bb3_2, %bb3_3) : (index, index, index, index) -> index
  // CHECK: return %[[OP3]]
  return %2 : index
}

// -----

// CHECK-LABEL: @foldCondBrArguments
// CHECK-SAME: (%[[COND:.+]]: i1, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index)
func.func @foldCondBrArguments(%cond: i1, %arg1: index, %arg2: index) -> index {
  // CHECK: cf.cond_br %[[COND]], ^bb1, ^bb2
  cf.cond_br %cond, ^bb1(%arg1, %arg2, %arg2 : index, index, index),
                 ^bb2(%arg1, %arg1, %arg2 : index, index, index)
  // CHECK: ^bb1:
^bb1(%bb1_0: index, %bb1_1: index, %bb1_2: index):
  // CHECK: %[[OP1:.+]] = "some.op1"(%[[ARG1]], %[[ARG2]], %[[ARG2]])
  %0 = "some.op1"(%bb1_0, %bb1_1, %bb1_2) : (index, index, index) -> index
  // CHECK: %[[OP1]]
  return %0 : index
  // CHECK: ^bb2:
^bb2(%bb2_0: index, %bb2_1: index, %bb2_2: index):
  // CHECK: %[[OP2:.+]] = "some.op2"(%[[ARG1]], %[[ARG1]], %[[ARG2]])
  %1 = "some.op2"(%bb2_0, %bb2_1, %bb2_2) : (index, index, index) -> index
  // CHECK: return %[[OP2]]
  return %1 : index
}

// -----

// CHECK-LABEL: @elideBranchOperands
// CHECK-SAME: (%[[ARG0:.+]]: index, %[[ARG1:.+]]: index)
func.func @elideBranchOperands(%arg0: index, %arg1: index) -> i32 {
  // CHECK-DAG: %[[C5I32:.+]] = arith.constant 5 : i32
  // CHECK-DAG: %[[C1I32:.+]] = arith.constant 1 : i32
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  %initialValue = arith.constant 5 : i32
  // CHECK: cf.br ^bb1(%[[C5I32]], %[[ARG0]] : i32, index)
  cf.br ^loopHeader(%initialValue, %arg0, %arg1 : i32, index, index)
  // CHECK: ^bb1(%[[BB1_ARG0:.+]]: i32, %[[BB1_ARG1:.+]]: index)
^loopHeader(%headerValue: i32, %counter: index, %headerMax: index):
  // CHECK: %[[CMP:.+]] = arith.cmpi slt, %[[BB1_ARG1]], %[[ARG1]]
  %lessThan = arith.cmpi slt, %counter, %headerMax : index
  // CHECK: cf.cond_br %[[CMP]], ^bb2, ^bb3
  cf.cond_br %lessThan, ^loopBody(%headerValue, %headerMax : i32, index),
                     ^exit(%headerValue: i32)
  // CHECK: ^bb2:
^loopBody(%bodyValue: i32, %bodyMax: index):
  %cst1_i32 = arith.constant 1 : i32
  // CHECK-DAG: %[[SUM:.+]] = arith.addi %[[BB1_ARG0]], %[[C1I32]]
  %newValue = arith.addi %bodyValue, %cst1_i32 : i32
  %cst1 = arith.constant 1 : index
  // CHECK-DAG: %[[NEXT:.+]] = arith.addi %[[BB1_ARG1]], %[[C1]]
  %newCounter = arith.addi %counter, %cst1 : index
  // CHECK: cf.br ^bb1(%[[SUM]], %[[NEXT]] : i32, index)
  cf.br ^loopHeader(%newValue, %newCounter, %bodyMax : i32, index, index)
  // CHECK: ^bb3:
^exit(%finalValue: i32):
  // CHECK: return %[[BB1_ARG0]]
  return %finalValue : i32
}

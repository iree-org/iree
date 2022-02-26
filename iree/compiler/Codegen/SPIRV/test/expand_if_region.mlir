// RUN: iree-opt -split-input-file -allow-unregistered-dialect -iree-spirv-expand-if-region %s | FileCheck %s

// Note: A pull request is open to upstream this pattern:
//   https://reviews.llvm.org/D117019
// After it lands, this pattern can be replaced.

// CHECK-LABEL: func @parent_region_only_contains_if_op
func @parent_region_only_contains_if_op(%cond: i1, %val: i32) -> i32 {
  %if = scf.if %cond -> i32 {
    scf.yield %val: i32
  } else {
    scf.yield %val: i32
  }
  return %if: i32
}

// CHECK: scf.if
// CHECK:   scf.yield
// CHECK: else
// CHECK:   scf.yield

// -----

// CHECK-LABEL: func @if_op_without_else_branch
func @if_op_without_else_branch(%cond: i1, %v0: i32, %v1: i32, %buffer: memref<i32>) {
  %add = arith.addi %v0, %v1 : i32
  scf.if %cond {
    memref.store %add, %buffer[] : memref<i32>
  }
  %mul = arith.muli %v0, %v1 : i32
  memref.store %mul, %buffer[] : memref<i32>
  return
}

// CHECK: arith.addi
// CHECK: scf.if
// CHECK:   memref.store
// CHECK: arith.muli
// CHECK: memref.store

// -----

// CHECK-LABEL: func @non_side_effecting_ops_before_and_after_if_op
//  CHECK-SAME: (%[[COND:.+]]: i1, %[[V0:.+]]: i32, %[[V1:.+]]: i32)
func @non_side_effecting_ops_before_and_after_if_op(%cond: i1, %v0: i32, %v1: i32) -> (i32, i32) {
  %add = arith.addi %v0, %v1 : i32
  %sub = arith.subi %v0, %v1 : i32
  %if = scf.if %cond -> i32 {
    scf.yield %add: i32
  } else {
    scf.yield %sub: i32
  }
  %mul = arith.muli %if, %v0 : i32
  %div = arith.divsi %if, %v1 : i32
  return %mul, %div: i32, i32
}

// CHECK: %[[IF:.+]]:2 = scf.if %[[COND]] -> (i32, i32)
// CHECK:   %[[ADD:.+]] = arith.addi %[[V0]], %[[V1]]
// CHECK:   %[[MUL:.+]] = arith.muli %[[ADD]], %[[V0]]
// CHECK:   %[[DIV:.+]] = arith.divsi %[[ADD]], %[[V1]]
// CHECK:   scf.yield %[[MUL]], %[[DIV]]
// CHECK: } else {
// CHECK:   %[[SUB:.+]] = arith.subi %[[V0]], %[[V1]]
// CHECK:   %[[MUL:.+]] = arith.muli %[[SUB]], %[[V0]]
// CHECK:   %[[DIV:.+]] = arith.divsi %[[SUB]], %[[V1]]
// CHECK:   scf.yield %[[MUL]], %[[DIV]]
// CHECK: return %[[IF]]#0, %[[IF]]#1

// -----

// CHECK-LABEL: func @side_effect_op_before_after_if_op
func @side_effect_op_before_after_if_op(%cond: i1, %v0: i32, %v1: i32, %buffer: memref<3xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  memref.store %v0, %buffer[%c0] : memref<3xi32>
  %add = arith.addi %v0, %v1 : i32
  scf.if %cond {
    memref.store %add, %buffer[%c1] : memref<3xi32>
  } else {
    memref.store %add, %buffer[%c1] : memref<3xi32>
  }
  %mul = arith.muli %v0, %v1 : i32
  memref.store %mul, %buffer[%c2] : memref<3xi32>
  return
}

// The control in the test pass allows moving side effecting ops.

// CHECK: scf.if
// CHECK:   memref.store
// CHECK:   arith.addi
// CHECK:   memref.store
// CHECK:   arith.muli
// CHECK:   memref.store
// CHECK: } else {
// CHECK:   memref.store
// CHECK:   arith.addi
// CHECK:   memref.store
// CHECK:   arith.muli
// CHECK:   memref.store

// -----

// CHECK-LABEL: func @zero_result_if_op
func @zero_result_if_op(%cond: i1, %v0: i32, %v1: i32, %buffer: memref<i32>) {
  %add = arith.addi %v0, %v1 : i32
  %sub = arith.subi %v0, %v1 : i32
  scf.if %cond {
    memref.store %add, %buffer[] : memref<i32>
  } else {
    memref.store %sub, %buffer[] : memref<i32>
  }
  %mul = arith.muli %v0, %v1 : i32
  memref.store %mul, %buffer[] : memref<i32>
  return
}

// CHECK: scf.if
// CHECK:   %[[ADD:.+]] = arith.addi
// CHECK:   memref.store %[[ADD]]
// CHECK:   %[[MUL:.+]] = arith.muli
// CHECK:   memref.store %[[MUL]]
// CHECK: } else {
// CHECK:   %[[SUB:.+]] = arith.subi
// CHECK:   memref.store %[[SUB]]
// CHECK:   %[[MUL:.+]] = arith.muli
// CHECK:   memref.store %[[MUL]]

// -----

// CHECK-LABEL: func @multi_result_if_op
//  CHECK-SAME: (%[[COND:.+]]: i1, %[[V0:.+]]: i32, %[[V1:.+]]: i32)
func @multi_result_if_op(%cond: i1, %v0: i32, %v1: i32) -> (i32, i32) {
  %add = arith.addi %v0, %v1 : i32
  %sub = arith.subi %v0, %v1 : i32
  %if:2 = scf.if %cond -> (i32, i32) {
    scf.yield %add, %sub: i32, i32
  } else {
    scf.yield %sub, %add: i32, i32
  }
  %mul = arith.muli %if#0, %v0 : i32
  %div = arith.divsi %if#1, %v1 : i32
  return %mul, %div: i32, i32
}

// CHECK: %[[IF:.+]]:2 = scf.if
// CHECK:   %[[ADD:.+]] = arith.addi %[[V0]], %[[V1]]
// CHECK:   %[[SUB:.+]] = arith.subi %[[V0]], %[[V1]]
// CHECK:   %[[MUL:.+]] = arith.muli %[[ADD]], %[[V0]]
// CHECK:   %[[DIV:.+]] = arith.divsi %[[SUB]], %[[V1]]
// CHECK:   scf.yield %[[MUL]], %[[DIV]]
// CHECK: } else {
// CHECK:   %[[ADD:.+]] = arith.addi %[[V0]], %[[V1]]
// CHECK:   %[[SUB:.+]] = arith.subi %[[V0]], %[[V1]]
// CHECK:   %[[MUL:.+]] = arith.muli %[[SUB]], %[[V0]]
// CHECK:   %[[DIV:.+]] = arith.divsi %[[ADD]], %[[V1]]
// CHECK:   scf.yield %[[MUL]], %[[DIV]]
// CHECK: return %[[IF]]#0, %[[IF]]#1

// -----

// CHECK-LABEL: func @multi_use_in_terminator
//  CHECK-SAME: (%[[COND:.+]]: i1, %[[V0:.+]]: i32, %[[V1:.+]]: i32)
func @multi_use_in_terminator(%cond: i1, %v0: i32, %v1: i32) -> (i32, i32, i32) {
  %add = arith.addi %v0, %v1 : i32
  %sub = arith.subi %v0, %v1 : i32
  %if = scf.if %cond -> i32 {
    scf.yield %add: i32
  } else {
    scf.yield %sub: i32
  }
  %mul = arith.muli %if, %if : i32
  return %mul, %mul, %mul: i32, i32, i32
}

// CHECK: %[[IF:.+]]:3 = scf.if
// CHECK:   %[[ADD:.+]] = arith.addi %[[V0]], %[[V1]]
// CHECK:   %[[MUL:.+]] = arith.muli %[[ADD]], %[[ADD]]
// CHECK:   scf.yield %[[MUL]], %[[MUL]], %[[MUL]]
// CHECK: } else {
// CHECK:   %[[SUB:.+]] = arith.subi %[[V0]], %[[V1]]
// CHECK:   %[[MUL:.+]] = arith.muli %[[SUB]], %[[SUB]]
// CHECK:   scf.yield %[[MUL]], %[[MUL]], %[[MUL]]
// CHECK: return %[[IF]]#0, %[[IF]]#1, %[[IF]]#2

// -----

// CHECK-LABEL: func @sticky_op_before_if_op
func @sticky_op_before_if_op(%cond: i1, %index: index, %v0: i32, %v1: i32, %buffer: memref<3xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %loc = arith.muli %index, %index : index
  memref.store %v0, %buffer[%loc] {sticky} : memref<3xi32>
  %add = arith.addi %v0, %v1 : i32
  scf.if %cond {
    memref.store %add, %buffer[%c1] : memref<3xi32>
  } else {
    memref.store %add, %buffer[%c1] : memref<3xi32>
  }
  %mul = arith.muli %v0, %v1 : i32
  memref.store %mul, %buffer[%c2] : memref<3xi32>
  return
}

// The control in the test pass disallows moving previous ops with "sticky" attribute and its backward slice.

// CHECK: arith.muli
// CHECK: memref.store {{.*}} {sticky}
// CHECK: scf.if
// CHECK:   arith.constant
// CHECK:   arith.constant
// CHECK:   arith.constant
// CHECK:   arith.addi
// CHECK:   memref.store
// CHECK:   arith.muli
// CHECK:   memref.store
// CHECK: } else {
// CHECK:   arith.constant
// CHECK:   arith.constant
// CHECK:   arith.constant
// CHECK:   arith.addi
// CHECK:   memref.store
// CHECK:   arith.muli
// CHECK:   memref.store

// -----

// CHECK-LABEL: func @sticky_op_after_if_op
func @sticky_op_after_if_op(%cond: i1, %index: index, %v0: i32, %v1: i32, %buffer: memref<3xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %loc = arith.muli %index, %index : index
  memref.store %v0, %buffer[%loc] : memref<3xi32>
  %add = arith.addi %v0, %v1 : i32
  scf.if %cond {
    memref.store %add, %buffer[%c1] : memref<3xi32>
  } else {
    memref.store %add, %buffer[%c1] : memref<3xi32>
  }
  %mul = arith.muli %v0, %v1 : i32
  memref.store %mul, %buffer[%c2] {sticky} : memref<3xi32>
  return
}

// NYI case for "sticky" next ops and its backward slice.

// CHECK: arith.muli
// CHECK: memref.store
// CHECK: arith.addi
// CHECK: scf.if {{.*}} {
// CHECK:   memref.store
// CHECK: } else {
// CHECK:   memref.store
// CHECK: }
// CHECK: arith.muli
// CHECK: memref.store {{.*}} {sticky}

// -----

// CHECK-LABEL: func @condition_back_slice
func @condition_back_slice(%cond0: i1, %cond1: i1, %cond2 : i1, %v0: i32, %v1: i32, %buffer: memref<3xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %add = arith.addi %v0, %v1 : i32
  %and = arith.andi %cond0, %cond1 : i1
  %or = arith.ori %and, %cond2 : i1
  scf.if %or {
    memref.store %add, %buffer[%c1] : memref<3xi32>
  } else {
    memref.store %add, %buffer[%c1] : memref<3xi32>
  }
  return
}

// CHECK: %[[AND:.+]] = arith.andi
// CHECK: %[[OR:.+]] = arith.ori %[[AND]]
// CHECK: scf.if %[[OR]]
// CHECK:   arith.constant
// CHECK:   arith.constant
// CHECK:   arith.constant
// CHECK:   arith.addi
// CHECK:   memref.store
// CHECK: } else {
// CHECK:   arith.constant
// CHECK:   arith.constant
// CHECK:   arith.constant
// CHECK:   arith.addi
// CHECK:   memref.store

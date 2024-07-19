// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @cmp_eq
// CHECK-SAME: (%[[LHS:.+]]: !util.buffer, %[[RHS:.+]]: !util.buffer)
util.func @cmp_eq(%lhs: !util.buffer, %rhs: !util.buffer) -> i1 {
  // CHECK: = util.cmp.eq %[[LHS]], %[[RHS]] : !util.buffer
  %result = util.cmp.eq %lhs, %rhs : !util.buffer
  util.return %result : i1
}

// -----

// CHECK-LABEL: @cmp_ne
// CHECK-SAME: (%[[LHS:.+]]: !util.buffer, %[[RHS:.+]]: !util.buffer)
util.func @cmp_ne(%lhs: !util.buffer, %rhs: !util.buffer) -> i1 {
  // CHECK: = util.cmp.ne %[[LHS]], %[[RHS]] : !util.buffer
  %result = util.cmp.ne %lhs, %rhs : !util.buffer
  util.return %result : i1
}

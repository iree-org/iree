// RUN: iree-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @already_canonical
util.func public @already_canonical(%arg0: index) -> index  {
  // CHECK: util.assume.int
  %0 = util.assume.int %arg0<umin=0> : index
  util.return %0 : index
}

// -----

// CHECK-LABEL: @elide_constant_assumption
util.func public @elide_constant_assumption() -> index  {
  %cst = arith.constant 1 : index
  %0 = util.assume.int %cst<umin=0> : index
  // CHECK: %[[CST:.*]] = arith.constant 1 : index
  // CHECK: util.return %[[CST]]
  util.return %0 : index
}

// -----

// CHECK-LABEL: @fold_exact_value
util.func public @fold_exact_value(%arg0: index) -> index  {
  %0 = util.assume.int %arg0<umin=123, umax=123> : index
  // CHECK: %[[CST:.*]] = arith.constant 123 : index
  // CHECK: util.return %[[CST]]
  util.return %0 : index
}

// -----

// CHECK-LABEL: @fold_exact_value_unnormalized
util.func public @fold_exact_value_unnormalized(%arg0: index) -> index  {
  %0 = util.assume.int %arg0[<umin=123, umax=123>, <umin=123, umax=123>] : index
  // CHECK: %[[CST:.*]] = arith.constant 123 : index
  // CHECK: util.return %[[CST]]
  util.return %0 : index
}

// -----

// CHECK-LABEL: @elide_multi_constant_assumption
util.func public @elide_multi_constant_assumption(%arg0: index, %arg1: index) -> index, index, index {
  %cst = arith.constant 1 : index
  // CHECK: %[[CST:.*]] = arith.constant 1 : index
  // CHECK: %[[ASSUME:.*]]:2 = util.assume.int
  // CHECK-NEXT: %arg0<udiv = 2>,
  // CHECK-NEXT: %arg1<udiv = 4>
  // CHECK-NEXT: : index, index
  %0:3 = util.assume.int %arg0<udiv=2>, %cst<umin=0>, %arg1<udiv=4> : index, index, index
  // CHECK: util.return %[[ASSUME]]#0, %[[CST]], %[[ASSUME]]#1
  util.return %0#0, %0#1, %0#2 : index, index, index
}

// -----

// CHECK-LABEL: @broadcast_duplicate_assumptions
util.func public @broadcast_duplicate_assumptions(%arg0: index) -> index  {
  // CHECK: util.assume.int %arg0<umin = 0>
  %0 = util.assume.int %arg0[<umin=0>, <umin=0>] : index
  util.return %0 : index
}

// -----

// CHECK-LABEL: @dedup_duplicate_operands
util.func public @dedup_duplicate_operands(%arg0: index) -> index, index {
  // CHECK: %[[ASSUME:.*]] = util.assume.int %arg0<umax = 2> : index
  %0:2 = util.assume.int %arg0[<umax=2>, <umax=2>], %arg0<umin=0> : index, index
  // CHECK: util.return %[[ASSUME]], %[[ASSUME]]
  util.return %0#0, %0#1 : index, index
}

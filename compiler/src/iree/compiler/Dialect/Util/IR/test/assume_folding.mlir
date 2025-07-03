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
  // CHECK: %[[ASSUME:.*]] = util.assume.int %arg0<umin = 0, umax = 2> : index
  %0:2 = util.assume.int %arg0[<umax=2>, <umax=2>], %arg0<umin=0> : index, index
  // CHECK: util.return %[[ASSUME]], %[[ASSUME]]
  util.return %0#0, %0#1 : index, index
}

// -----

// CHECK-LABEL: @dedup_operands_both_single_range
util.func public @dedup_operands_both_single_range(%arg0: index) -> index, index {
  // CHECK: %[[ASSUME:.*]] = util.assume.int %arg0<umin = 9, umax = 200, udiv = 18> : index
  %0:2 = util.assume.int %arg0<umax=200, udiv = 9>, %arg0<umin=9, udiv = 6> : index, index
  // CHECK: util.return %[[ASSUME]], %[[ASSUME]]
  util.return %0#0, %0#1 : index, index
}

// -----

// CHECK-LABEL: @dedup_operands_first_single_range
util.func public @dedup_operands_first_single_range(%arg0: index) -> index, index {
  // CHECK: %[[ASSUME:.*]] = util.assume.int %arg0[<umin = 1, umax = 5>, <umin = 2, umax = 5>, <umin = 3, umax = 5>] : index
  %0:2 = util.assume.int %arg0<umax=5>, %arg0[<umin=1>, <umin=2>, <umin=3>] : index, index
  // CHECK: util.return %[[ASSUME]], %[[ASSUME]]
  util.return %0#0, %0#1 : index, index
}

// -----

// CHECK-LABEL: @dedup_operands_both_long
util.func public @dedup_operands_both_long(%arg0: index) -> index, index {
  // CHECK: %[[ASSUME:.*]] = util.assume.int %arg0[<umin = 1, umax = 5>, <umin = 2, umax = 4>] : index
  %0:2 = util.assume.int %arg0[<umax=5>, <umax=6>], %arg0[<umin=1, umax=7>, <umin=2, umax = 4>] : index, index
  // CHECK: util.return %[[ASSUME]], %[[ASSUME]]
  util.return %0#0, %0#1 : index, index
}

// -----

// CHECK-LABEL: @dedup_operands_multiple_groups
util.func public @dedup_operands_multiple_groups(%arg0: index, %arg1: index) -> index, index, index, index {
  // CHECK: %[[ASSUME:.*]]:2 = util.assume.int
  // CHECK-NEXT: %arg0[<umin = 2>, <umin = 4, umax = 4>],
  // CHECK-NEXT: %arg1<udiv = 10>
  %0:4 = util.assume.int
    %arg0[<umin=2>, <umin=4>],
    %arg1[<udiv=5>, <udiv=10>],
    %arg0[<umin=1>, <umin=2, umax = 4>],
    %arg1<udiv=2> : index, index, index, index
  // CHECK: util.return %[[ASSUME]]#0, %[[ASSUME]]#1, %[[ASSUME]]#0, %[[ASSUME]]#1
  util.return %0#0, %0#1, %0#2, %0#3 : index, index, index, index
}

// -----

// CHECK-LABEL: @dedup_operands_multiple_groups_and_singleton
util.func public @dedup_operands_multiple_groups_and_singleton(
  %arg0: index, %arg1: index, %arg2: index) -> index, index, index, index, index {
  // CHECK: %[[ASSUME:.*]]:3 = util.assume.int
  // CHECK-NEXT: %arg2<udiv = 64>,
  // CHECK-NEXT: %arg0<umin = 3>,
  // CHECK-NEXT: %arg1<umax = 5>
  %0:5 = util.assume.int
    %arg2<udiv=64>,
    %arg0<umin=2>,
    %arg1<umax=5>,
    %arg0<umin=3>,
    %arg1<umax=6> : index, index, index, index, index
  // CHECK: util.return %[[ASSUME]]#0, %[[ASSUME]]#1, %[[ASSUME]]#2, %[[ASSUME]]#1, %[[ASSUME]]#2
  util.return %0#0, %0#1, %0#2, %0#3, %0#4 : index, index, index, index, index
}

// -----

// CHECK-LABEL: @dedup_operands_three
util.func public @dedup_operands_three(%arg0: index) -> index, index, index {
  // CHECK: %[[ASSUME:.*]] = util.assume.int %arg0<umin = 3>
  %0:3 = util.assume.int
    %arg0<umin=1>,
    %arg0<umin=2>,
    %arg0<umin=3> : index, index, index
  // CHECK: util.return %[[ASSUME]], %[[ASSUME]], %[[ASSUME]]
  util.return %0#0, %0#1, %0#2 : index, index, index
}

// -----

// CHECK-LABEL: @fold_assume_of_div_mul
util.func public @fold_assume_of_div_mul(%arg0: index) -> index  {
  // CHECK-SAME: %[[ARG0:.+]]: index
  %0 = util.assume.int %arg0<udiv=4> : index
  // CHECK: %[[ASSUME:.*]] = util.assume.int %[[ARG0]]
  %c4 = arith.constant 4 : index
  %1 = arith.divui %0, %c4 : index
  %2 = arith.muli %1, %c4 : index
  // CHECK: util.return %[[ASSUME]]
  util.return %2 : index
}

// -----

// CHECK-LABEL: @fold_assume_of_div_mul_multiple
util.func public @fold_assume_of_div_mul_multiple(%arg0: index) -> index  {
  // CHECK-SAME: %[[ARG0:.+]]: index
  %0 = util.assume.int %arg0<udiv=8> : index
  // CHECK: %[[ASSUME:.*]] = util.assume.int %[[ARG0]]
  %c4 = arith.constant 4 : index
  %1 = arith.divui %0, %c4 : index
  %2 = arith.muli %1, %c4 : index
  // CHECK: util.return %[[ASSUME]]
  util.return %2 : index
}

// -----

// CHECK-LABEL: @nofold_assume_of_div_mul_indivisible
util.func public @nofold_assume_of_div_mul_indivisible(%arg0: index) -> index  {
  // CHECK-SAME: %[[ARG0:.+]]: index
  %0 = util.assume.int %arg0<udiv=7> : index
  // CHECK: %[[ASSUME:.*]] = util.assume.int %[[ARG0]]
  %c4 = arith.constant 4 : index
  %1 = arith.divui %0, %c4 : index
  // CHECK: %[[DIV:.*]] = arith.divui %[[ASSUME]]
  %2 = arith.muli %1, %c4 : index
  // CHECK: %[[MUL:.*]] = arith.muli %[[DIV]]
  // CHECK: util.return %[[MUL]]
  util.return %2 : index
}

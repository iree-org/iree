// RUN: iree-opt --split-input-file --iree-util-apply-patterns --allow-unregistered-dialect %s | FileCheck %s

// Tests that scf.if with both branches unreachable and results is replaced with
// poison values.

// CHECK-LABEL: @simplifyIfBothUnreachableWithResult
util.func public @simplifyIfBothUnreachableWithResult(%cond: i1) -> i32 {
  // CHECK-NOT: scf.if
  // CHECK: util.unreachable "both branches unreachable"
  %result = scf.if %cond -> i32 {
    util.scf.unreachable "then branch"
    %poison = ub.poison : i32
    scf.yield %poison : i32
  } else {
    util.scf.unreachable "else branch"
    %poison = ub.poison : i32
    scf.yield %poison : i32
  }
  util.return %result : i32
}

// -----

// Tests that scf.if with then branch unreachable inlines the else branch.

// CHECK-LABEL: @simplifyIfThenUnreachable
util.func public @simplifyIfThenUnreachable(%cond: i1) -> i32 {
  // CHECK-NOT: scf.if
  // CHECK: %[[C42:.*]] = arith.constant 42
  // CHECK: util.return %[[C42]]
  %result = scf.if %cond -> i32 {
    util.scf.unreachable "then branch"
    %poison = ub.poison : i32
    scf.yield %poison : i32
  } else {
    %c42 = arith.constant 42 : i32
    scf.yield %c42 : i32
  }
  util.return %result : i32
}

// -----

// Tests that scf.if with else branch unreachable inlines the then branch.

// CHECK-LABEL: @simplifyIfElseUnreachable
util.func public @simplifyIfElseUnreachable(%cond: i1) -> i32 {
  // CHECK-NOT: scf.if
  // CHECK: %[[C24:.*]] = arith.constant 24
  // CHECK: util.return %[[C24]]
  %result = scf.if %cond -> i32 {
    %c24 = arith.constant 24 : i32
    scf.yield %c24 : i32
  } else {
    util.scf.unreachable "else branch"
    %poison = ub.poison : i32
    scf.yield %poison : i32
  }
  util.return %result : i32
}

// -----

// Tests that scf.if with both branches unreachable and no results is replaced
// with unreachable.

// CHECK-LABEL: @simplifyIfBothUnreachable
util.func public @simplifyIfBothUnreachable(%cond: i1) {
  // CHECK-NOT: scf.if
  // CHECK: util.unreachable "both branches unreachable"
  scf.if %cond {
    util.scf.unreachable "then branch"
    scf.yield
  } else {
    util.scf.unreachable "else branch"
    scf.yield
  }
  util.return
}

// -----

// Tests that scf.if with side effects in reachable branch preserves them when
// inlining.

util.func private @must_be_preserved()
util.func private @can_be_dropped() -> i32

// CHECK-LABEL: @simplifyIfThenUnreachablePreserveSideEffects
util.func public @simplifyIfThenUnreachablePreserveSideEffects(%cond: i1) -> i32 {
  // CHECK-NOT: scf.if
  // CHECK: util.call @must_be_preserved()
  // CHECK: %[[VAL:.*]] = util.call @can_be_dropped()
  // CHECK: util.return %[[VAL]]
  %result = scf.if %cond -> i32 {
    util.scf.unreachable "then branch"
    %poison = ub.poison : i32
    scf.yield %poison : i32
  } else {
    util.call @must_be_preserved() : () -> ()
    %val = util.call @can_be_dropped() : () -> i32
    scf.yield %val : i32
  }
  util.return %result : i32
}

// -----

// Tests that scf.while with unreachable body returns init values.

// CHECK-LABEL: @simplifyWhileBodyUnreachable
util.func public @simplifyWhileBodyUnreachable(%init: i32) -> i32 {
  // CHECK-NOT: scf.while
  // CHECK: util.return %arg0
  %result = scf.while (%arg = %init) : (i32) -> i32 {
    %cond = "test.condition"(%arg) : (i32) -> i1
    scf.condition(%cond) %arg : i32
  } do {
  ^bb0(%arg: i32):
    util.scf.unreachable "body unreachable"
    scf.yield %arg : i32
  }
  util.return %result : i32
}

// -----

// Tests that scf.while with unreachable body and no results is simplified.

// CHECK-LABEL: @simplifyWhileBodyUnreachableNoResult
util.func public @simplifyWhileBodyUnreachableNoResult(%init: i32) {
  // CHECK-NOT: scf.while
  // CHECK: util.return
  %0 = scf.while (%arg = %init) : (i32) -> (i32) {
    %cond = "test.condition"(%arg) : (i32) -> i1
    scf.condition(%cond) %arg : i32
  } do {
  ^bb0(%arg: i32):
    util.scf.unreachable "body unreachable"
    scf.yield %arg : i32
  }
  util.return
}

// -----

// Tests that scf.for with unreachable body returns iter_args.

// CHECK-LABEL: @simplifyForBodyUnreachable
util.func public @simplifyForBodyUnreachable(%lb: index, %ub: index, %step: index, %init: i32) -> i32 {
  // CHECK-NOT: scf.for
  // CHECK: util.return %arg3
  %result = scf.for %i = %lb to %ub step %step iter_args(%arg = %init) -> i32 {
    util.scf.unreachable "loop body"
    scf.yield %arg : i32
  }
  util.return %result : i32
}

// -----

// Tests that scf.for with unreachable body and no iter_args is simplified.

// CHECK-LABEL: @simplifyForBodyUnreachableNoResult
util.func public @simplifyForBodyUnreachableNoResult(%lb: index, %ub: index, %step: index) {
  // CHECK-NOT: scf.for
  // CHECK: util.return
  scf.for %i = %lb to %ub step %step {
    util.scf.unreachable "loop body"
    scf.yield
  }
  util.return
}

// -----

// Tests that scf.index_switch with all cases unreachable is replaced with
// poison values.

// CHECK-LABEL: @simplifyIndexSwitchAllUnreachable
util.func public @simplifyIndexSwitchAllUnreachable(%idx: index) -> i32 {
  // CHECK-NOT: scf.index_switch
  // CHECK: util.unreachable
  %result = scf.index_switch %idx -> i32
  case 0 {
    util.scf.unreachable "case 0"
    %poison = ub.poison : i32
    scf.yield %poison : i32
  }
  case 1 {
    util.scf.unreachable "case 1"
    %poison = ub.poison : i32
    scf.yield %poison : i32
  }
  default {
    util.scf.unreachable "default"
    %poison = ub.poison : i32
    scf.yield %poison : i32
  }
  util.return %result : i32
}

// -----

// Tests that scf.index_switch with some unreachable cases removes only those
// cases.

// CHECK-LABEL: @simplifyIndexSwitchCaseUnreachable
util.func public @simplifyIndexSwitchCaseUnreachable(%idx: index) -> i32 {
  // Constants may be hoisted out:
  // CHECK-DAG: %[[C88:.*]] = arith.constant 88
  // CHECK-DAG: %[[C42:.*]] = arith.constant 42
  // CHECK-DAG: %[[C24:.*]] = arith.constant 24
  // CHECK: scf.index_switch %arg0 -> i32
  // CHECK-NOT: case 0
  // CHECK: case 1 {
  // CHECK:   "test.side_effect"
  // CHECK:   scf.yield %[[C42]]
  // CHECK: }
  // CHECK: case 2 {
  // CHECK:   "test.side_effect"
  // CHECK:   scf.yield %[[C88]]
  // CHECK: }
  // CHECK: default {
  // CHECK:   "test.side_effect"
  // CHECK:   scf.yield %[[C24]]
  // CHECK: }
  %result = scf.index_switch %idx -> i32
  case 0 {
    util.scf.unreachable "case 0"
    %poison = ub.poison : i32
    scf.yield %poison : i32
  }
  case 1 {
    "test.side_effect"() : () -> ()
    %c42 = arith.constant 42 : i32
    scf.yield %c42 : i32
  }
  case 2 {
    "test.side_effect"() : () -> ()
    %c88 = arith.constant 88 : i32
    scf.yield %c88 : i32
  }
  default {
    "test.side_effect"() : () -> ()
    %c24 = arith.constant 24 : i32
    scf.yield %c24 : i32
  }
  util.return %result : i32
}

// -----

// Tests that scf.index_switch with only default remaining is inlined.

// CHECK-LABEL: @simplifyIndexSwitchOnlyDefaultRemains
util.func public @simplifyIndexSwitchOnlyDefaultRemains(%idx: index) -> i32 {
  // CHECK-NOT: scf.index_switch
  // CHECK: %[[C99:.*]] = arith.constant 99
  // CHECK: util.return %[[C99]]
  %result = scf.index_switch %idx -> i32
  case 0 {
    util.scf.unreachable "case 0"
    %poison = ub.poison : i32
    scf.yield %poison : i32
  }
  case 1 {
    util.scf.unreachable "case 1"
    %poison = ub.poison : i32
    scf.yield %poison : i32
  }
  default {
    %c99 = arith.constant 99 : i32
    scf.yield %c99 : i32
  }
  util.return %result : i32
}

// -----

// Tests nested scf.if with inner unreachable branches.

// CHECK-LABEL: @simplifyNestedIfInnerUnreachable
util.func public @simplifyNestedIfInnerUnreachable(%cond1: i1, %cond2: i1) -> i32 {
  // The inner if is simplified, and the outer may be optimized to select
  // with side effects hoisted.
  // CHECK-DAG: %[[C24:.*]] = arith.constant 24
  // CHECK-DAG: %[[C42:.*]] = arith.constant 42
  // CHECK: arith.select %arg0, %[[C42]], %[[C24]]
  // CHECK: scf.if %arg0 {
  // CHECK:   "test.side_effect"
  // CHECK: } else {
  // CHECK:   "test.side_effect"
  // CHECK: }
  %result = scf.if %cond1 -> i32 {
    %inner = scf.if %cond2 -> i32 {
      util.scf.unreachable "inner then"
      %poison = ub.poison : i32
      scf.yield %poison : i32
    } else {
      "test.side_effect"() : () -> ()
      %c42 = arith.constant 42 : i32
      scf.yield %c42 : i32
    }
    scf.yield %inner : i32
  } else {
    "test.side_effect"() : () -> ()
    %c24 = arith.constant 24 : i32
    scf.yield %c24 : i32
  }
  util.return %result : i32
}

// -----

// Tests scf.if inside scf.for with unreachable branch.

// CHECK-LABEL: @simplifyIfInForWithUnreachable
util.func public @simplifyIfInForWithUnreachable(%lb: index, %ub: index, %step: index) -> i32 {
  %c0 = arith.constant 0 : i32
  // CHECK: scf.for
  // CHECK-NOT: scf.if
  // CHECK:   %[[VAL:.*]] = "test.produce"
  // CHECK:   scf.yield %[[VAL]]
  %result = scf.for %i = %lb to %ub step %step iter_args(%arg = %c0) -> i32 {
    %cond = "test.condition"(%i) : (index) -> i1
    %val = scf.if %cond -> i32 {
      util.scf.unreachable "then unreachable"
      %poison = ub.poison : i32
      scf.yield %poison : i32
    } else {
      %v = "test.produce"(%arg) : (i32) -> i32
      scf.yield %v : i32
    }
    scf.yield %val : i32
  }
  util.return %result : i32
}

// -----

// Tests handling of multiple result types with unreachable.

// CHECK-LABEL: @simplifyIfMultipleResultTypes
util.func public @simplifyIfMultipleResultTypes(%cond: i1) -> (i32, f32, tensor<4xf32>) {
  // CHECK-NOT: scf.if
  // CHECK: util.unreachable "both branches unreachable"
  %result:3 = scf.if %cond -> (i32, f32, tensor<4xf32>) {
    util.scf.unreachable "then"
    %p1 = ub.poison : i32
    %p2 = ub.poison : f32
    %p3 = ub.poison : tensor<4xf32>
    scf.yield %p1, %p2, %p3 : i32, f32, tensor<4xf32>
  } else {
    util.scf.unreachable "else"
    %p1 = ub.poison : i32
    %p2 = ub.poison : f32
    %p3 = ub.poison : tensor<4xf32>
    scf.yield %p1, %p2, %p3 : i32, f32, tensor<4xf32>
  }
  util.return %result#0, %result#1, %result#2 : i32, f32, tensor<4xf32>
}

// -----

// Tests that a cond_br with the true branch unreachable is converted to an
// unconditional br to the false target.

// CHECK-LABEL: @simplifyCondBrTrueUnreachable
// CHECK-SAME: (%[[COND:.+]]: i1)
util.func @simplifyCondBrTrueUnreachable(%cond: i1) -> i32 {
  // CHECK-NOT: cf.cond_br
  // CHECK-NOT: ^bb
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  util.unreachable "true branch"
^bb2:
  // CHECK: %[[RESULT:.+]] = "some.op"() : () -> i32
  %result = "some.op"() : () -> i32
  // CHECK: util.return %[[RESULT]]
  util.return %result : i32
}

// -----

// Tests that a cond_br with the false branch unreachable is converted to an
// unconditional br to the true target.

// CHECK-LABEL: @simplifyCondBrFalseUnreachable
// CHECK-SAME: (%[[COND:.+]]: i1)
util.func @simplifyCondBrFalseUnreachable(%cond: i1) -> i32 {
  // CHECK-NOT: cf.cond_br
  // CHECK-NOT: ^bb
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  // CHECK: %[[RESULT:.+]] = "some.op"() : () -> i32
  %result = "some.op"() : () -> i32
  // CHECK: util.return %[[RESULT]]
  util.return %result : i32
^bb2:
  util.unreachable "false branch"
}

// -----

// Tests that a cond_br with both branches unreachable is replaced with
// unreachable.

// CHECK-LABEL: @simplifyCondBrBothUnreachable
// CHECK-SAME: (%[[COND:.+]]: i1)
util.func @simplifyCondBrBothUnreachable(%cond: i1) {
  // CHECK-NOT: cf.cond_br
  // CHECK: util.unreachable "both branches unreachable"
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  util.unreachable "true"
^bb2:
  util.unreachable "false"
}

// -----

// Tests that side-effect-free ops in the same block before unreachable are
// removed (side-effecting ops may be infinite loops and we can't kill them).

// CHECK-LABEL: @removeOpsBeforeUnreachable
util.func @removeOpsBeforeUnreachable() {
  // CHECK-NOT: arith.constant
  // CHECK-NOT: arith.addi
  // CHECK-NOT: arith.muli
  // CHECK: util.unreachable
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %sum = arith.addi %c1, %c2 : i32
  %prod = arith.muli %sum, %c2 : i32
  util.unreachable
}

// -----

// Tests that ops with side effects are preserved before unreachable.

// CHECK-LABEL: @dontRemoveOpsWithSideEffects
util.func @dontRemoveOpsWithSideEffects() {
  // CHECK: "some.side.effect"
  "some.side.effect"() : () -> ()
  // CHECK-NOT: arith.constant
  %c1 = arith.constant 1 : i32
  // CHECK: util.unreachable
  util.unreachable
}

// -----

// Tests that ops with external uses are not removed and the pattern correctly
// preserves %c1 since it's used as a branch operand.

// CHECK-LABEL: @dontRemoveOpsWithExternalUses
// CHECK-SAME: (%[[COND:.+]]: i1)
util.func @dontRemoveOpsWithExternalUses(%cond: i1) -> i32 {
  // CHECK: %[[C1:.+]] = arith.constant 1 : i32
  // CHECK-NOT: cf.cond_br
  // CHECK-NOT: ^bb
  %c1 = arith.constant 1 : i32
  cf.cond_br %cond, ^bb1(%c1 : i32), ^bb2
^bb1(%arg: i32):
  // CHECK: util.return %[[C1]]
  util.return %arg : i32
^bb2:
  // Dead ops here can be removed since no external uses.
  %c2 = arith.constant 2 : i32
  util.unreachable
}

// -----

// Tests that a full CFG with unreachable branches gets completely optimized
// away through pattern composition (CFG simplification + DCE).

// CHECK-LABEL: @compositionCondBrTrueUnreachable
// CHECK-SAME: (%[[COND:.+]]: i1)
util.func @compositionCondBrTrueUnreachable(%cond: i1) -> i32 {
  // CHECK-NOT: cf.cond_br
  // CHECK-NOT: ^bb
  // CHECK: %[[RESULT:.+]] = "false.side.effect"() : () -> i32
  // CHECK: util.return %[[RESULT]]
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  util.unreachable "true branch"
^bb2:
  %result = "false.side.effect"() : () -> i32
  util.return %result : i32
}

// -----

// Tests that unreachable causes dead code elimination of ops before it.

// CHECK-LABEL: @compositionCondBrWithDeadCode
util.func @compositionCondBrWithDeadCode(%cond: i1) {
  // CHECK-NOT: arith.constant
  // CHECK-NOT: cf.cond_br
  // CHECK: util.unreachable
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  util.unreachable
^bb2:
  util.unreachable
}

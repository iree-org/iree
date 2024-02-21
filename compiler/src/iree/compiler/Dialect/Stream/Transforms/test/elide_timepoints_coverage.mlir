// RUN: iree-opt --split-input-file --iree-stream-elide-timepoints %s | FileCheck %s

// Tests that we don't (currently) do anything with global forwarding.
// Generic util analysis passes operating on globals can do things like folding.
// We just want to make sure here that we are preserving the global behavior.

util.global private mutable @global0 : !stream.timepoint
util.global private mutable @global1 : !stream.timepoint

util.initializer {
  %t0 = stream.cmd.execute with() {} => !stream.timepoint
  util.global.store %t0, @global0 : !stream.timepoint
  %t1 = stream.cmd.execute await(%t0) => with() {} => !stream.timepoint
  util.global.store %t1, @global1 : !stream.timepoint
  util.return
}

// CHECK-LABEL: @initializedGlobals
util.func private @initializedGlobals() -> !stream.timepoint {
  // CHECK: %[[GLOBAL0:.+]] = util.global.load @global0
  %global0 = util.global.load @global0 : !stream.timepoint
  // CHECK: %[[GLOBAL1:.+]] = util.global.load @global1
  %global1 = util.global.load @global1 : !stream.timepoint
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[GLOBAL0]], %[[GLOBAL1]])
  %join = stream.timepoint.join max(%global0, %global1) => !stream.timepoint
  // CHECK: util.return %[[JOIN]]
  util.return %join : !stream.timepoint
}

// -----

// Tests that meaningful timeline ops are never marked immediate.

// CHECK-LABEL: @nonImmediate
util.func private @nonImmediate() -> !stream.timepoint {
  // CHECK: %[[EXECUTE:.+]] = stream.cmd.execute
  %0 = stream.cmd.execute with() {} => !stream.timepoint
  // CHECK: util.return %[[EXECUTE]]
  util.return %0 : !stream.timepoint
}

// -----

// Tests that coverage propagates through timeline ops. Here %exec0 is covered
// by both %exec1a and %exec1b and does not need to be joined.

// CHECK-LABEL: @joinChained
util.func public @joinChained() -> !stream.timepoint {
  // CHECK: %[[EXEC0:.+]] = stream.cmd.execute with
  %exec0 = stream.cmd.execute with() {} => !stream.timepoint
  // CHECK: %[[EXEC1A:.+]] = stream.cmd.execute await(%[[EXEC0]])
  %exec1a = stream.cmd.execute await(%exec0) => with() {} => !stream.timepoint
  // CHECK: %[[EXEC1B:.+]] = stream.cmd.execute await(%[[EXEC0]])
  %exec1b = stream.cmd.execute await(%exec0) => with() {} => !stream.timepoint
  // CHECK: %[[EXEC0_IMM:.+]] = stream.timepoint.immediate
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[EXEC0_IMM]], %[[EXEC1A]], %[[EXEC1B]])
  %join = stream.timepoint.join max(%exec0, %exec1a, %exec1b) => !stream.timepoint
  // CHECK: util.return %[[JOIN]]
  util.return %join : !stream.timepoint
}

// -----

// Tests that coverage propagates through a select: %exec0 is covered by both
// the true and false conditions and does not need to be joined.

// CHECK-LABEL: @selectCovered
// CHECK-SAME: (%[[COND:.+]]: i1)
util.func public @selectCovered(%cond: i1) -> !stream.timepoint {
  // CHECK: %[[EXEC0:.+]] = stream.cmd.execute
  %exec0 = stream.cmd.execute with() {} => !stream.timepoint
  // CHECK: %[[EXEC1A:.+]] = stream.cmd.execute await(%[[EXEC0]])
  %exec1a = stream.cmd.execute await(%exec0) => with() {} => !stream.timepoint
  // CHECK: %[[EXEC1B:.+]] = stream.cmd.execute await(%[[EXEC0]])
  %exec1b = stream.cmd.execute await(%exec0) => with() {} => !stream.timepoint
  // CHECK: %[[SELECT:.+]] = arith.select %[[COND]], %[[EXEC1A]], %[[EXEC1B]]
  %select = arith.select %cond, %exec1a, %exec1b : !stream.timepoint
  // CHECK: %[[EXEC0_IMM:.+]] = stream.timepoint.immediate
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[EXEC0_IMM]], %[[SELECT]])
  %join = stream.timepoint.join max(%exec0, %select) => !stream.timepoint
  // CHECK: util.return %[[JOIN]]
  util.return %join : !stream.timepoint
}

// -----

// Tests that a timepoint passed along a call edge is propagated.
// %t0/%t1 are covered by the call result %call that joins the two together.

// CHECK-LABEL: util.func public @caller
// CHECK-SAME: (%[[T0:.+]]: !stream.timepoint, %[[T1:.+]]: !stream.timepoint)
util.func public @caller(%t0: !stream.timepoint, %t1: !stream.timepoint) -> !stream.timepoint {
  // CHECK: %[[CALL:.+]] = util.call @callee(%[[T0]], %[[T1]])
  %call = util.call @callee(%t0, %t1) : (!stream.timepoint, !stream.timepoint) -> !stream.timepoint
  // CHECK-DAG: %[[T0_COVERED:.+]] = stream.timepoint.immediate
  // CHECK-DAG: %[[T1_COVERED:.+]] = stream.timepoint.immediate
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[T0_COVERED]], %[[T1_COVERED]], %[[CALL]])
  %join = stream.timepoint.join max(%t0, %t1, %call) => !stream.timepoint
  // CHECK: util.return %[[JOIN]]
  util.return %join : !stream.timepoint
}
// CHECK-LABEL: util.func private @callee
util.func private @callee(%t0a: !stream.timepoint, %t0b: !stream.timepoint) -> !stream.timepoint {
  // CHECK-NOT: stream.timepoint.immediate
  // CHECK: %[[JOIN_CALLEE:.+]] = stream.timepoint.join max
  %t1 = stream.timepoint.join max(%t0a, %t0b) => !stream.timepoint
  // CHECK: util.return %[[JOIN_CALLEE]]
  util.return %t1 : !stream.timepoint
}

// -----

// Tests that duplicate call args/results are handled correctly.
// Ideally we're running in as part of a fixed-point iteration with IPO that
// removes the dupes and lets us focus on simpler cases. For now we don't do
// anything clever with folding the call results even though we know they're
// the same and instead just handle coverage (hitting either call results is
// the same as hitting the original arg).

// CHECK-LABEL: util.func public @callerDupes
util.func public @callerDupes(%unknown: !stream.timepoint) -> !stream.timepoint {
  // CHECK: %[[CALL:.+]]:2 = util.call @calleeDupes
  %call:2 = util.call @calleeDupes(%unknown, %unknown) : (!stream.timepoint, !stream.timepoint) -> (!stream.timepoint, !stream.timepoint)
  // CHECK-NEXT: %[[UNKNOWN_IMM:.+]] = stream.timepoint.immediate
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[UNKNOWN_IMM]], %[[CALL]]#0, %[[CALL]]#1)
  %join = stream.timepoint.join max(%unknown, %call#0, %call#1) => !stream.timepoint
  // CHECK: util.return %[[JOIN]]
  util.return %join : !stream.timepoint
}
util.func private @calleeDupes(%arg0: !stream.timepoint, %arg1: !stream.timepoint) -> (!stream.timepoint, !stream.timepoint) {
  util.return %arg0, %arg1 : !stream.timepoint, !stream.timepoint
}

// -----

// Tests that calls with non-uniform args still track partial coverage.
// Here the result of @nonUniformCallee always covers %t0 but not %t1 and we're
// able to elide %t0 in the final join.

// TODO(benvanik): we should also be able to trim the calls/t1 and only use
// %t01 but that needs some work to know that call0 == t0 and call1 == t01.

// CHECK-LABEL: util.func public @nonUniformCaller
// CHECK-SAME: (%[[T0:.+]]: !stream.timepoint, %[[T1:.+]]: !stream.timepoint)
util.func public @nonUniformCaller(%t0: !stream.timepoint, %t1: !stream.timepoint) -> !stream.timepoint {
  // CHECK: %[[CALL0:.+]] = util.call @nonUniformCallee(%[[T0]])
  %call0 = util.call @nonUniformCallee(%t0) : (!stream.timepoint) -> !stream.timepoint
  // CHECK: %[[T01:.+]] = stream.timepoint.join max(%[[T0]], %[[T1]])
  %t01 = stream.timepoint.join max(%t0, %t1) => !stream.timepoint
  // CHECK: %[[CALL1:.+]] = util.call @nonUniformCallee(%[[T01]])
  %call1 = util.call @nonUniformCallee(%t01) : (!stream.timepoint) -> !stream.timepoint
  // CHECK: %[[T0_IMM:.+]] = stream.timepoint.immediate
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[T0_IMM]], %[[CALL0]], %[[T1]], %[[CALL1]])
  %join = stream.timepoint.join max(%t0, %call0, %t1, %call1) => !stream.timepoint
  // CHECK: util.return %[[JOIN]]
  util.return %join : !stream.timepoint
}
// CHECK: util.func private @nonUniformCallee
util.func private @nonUniformCallee(%arg0: !stream.timepoint) -> !stream.timepoint {
  util.return %arg0 : !stream.timepoint
}

// -----

// Tests that timepoints are tracked through branches args.
// In this simple case %bb1_t0 always covers %t0.

// CHECK-LABEL: util.func public @branch
// CHECK-SAME: (%[[T0:.+]]: !stream.timepoint)
util.func public @branch(%t0: !stream.timepoint) -> !stream.timepoint {
  // CHECK: cf.br ^bb1
  cf.br ^bb1(%t0 : !stream.timepoint)
// CHECK-NEXT: ^bb1(%[[BB1_T0:.+]]: !stream.timepoint)
^bb1(%bb1_t0: !stream.timepoint):
  // CHECK: %[[T0_IMM:.+]] = stream.timepoint.immediate
  // CHECK-NEXT: %[[JOIN:.+]] = stream.timepoint.join max(%[[T0_IMM]], %[[BB1_T0]])
  %join = stream.timepoint.join max(%t0, %bb1_t0) => !stream.timepoint
  // CHECK: util.return %[[JOIN]]
  util.return %join : !stream.timepoint
}

// -----

// Tests that forward edges with convergent timepoints track coverage.
// Here both true and false paths cover %t0 and it can be elided at the join.

// CHECK-LABEL: util.func public @branchConvergentForwardEdge
// CHECK-SAME: (%[[COND:.+]]: i1, %[[T0:.+]]: !stream.timepoint)
util.func public @branchConvergentForwardEdge(%cond: i1, %t0: !stream.timepoint) -> !stream.timepoint {
  // CHECK: %[[T1A:.+]] = stream.cmd.execute await(%[[T0]])
  %t1a = stream.cmd.execute await(%t0) => with() {} => !stream.timepoint
  // CHECK: %[[T1B:.+]] = stream.cmd.execute await(%[[T0]])
  %t1b = stream.cmd.execute await(%t0) => with() {} => !stream.timepoint
  // CHECK: cf.cond_br %[[COND]]
  // CHECK-SAME:   ^bb1(%[[T1A]] : !stream.timepoint),
  // CHECK-SAME:   ^bb1(%[[T1B]] : !stream.timepoint)
  cf.cond_br %cond, ^bb1(%t1a : !stream.timepoint), ^bb1(%t1b : !stream.timepoint)
// CHECK-NEXT: ^bb1(%[[BB1_ARG:.+]]: !stream.timepoint)
^bb1(%bb1_arg: !stream.timepoint):
  // CHECK: %[[T0_IMM:.+]] = stream.timepoint.immediate
  // CHECK-NEXT: %[[JOIN:.+]] = stream.timepoint.join max(%[[T0_IMM]], %[[BB1_ARG]])
  %join = stream.timepoint.join max(%t0, %bb1_arg) => !stream.timepoint
  // CHECK: util.return %[[JOIN]]
  util.return %join : !stream.timepoint
}

// -----

// Tests that forward edges with divergent timepoint coverage get propagated.
// %t0 is covered on both paths but %t1 is only covered when %cond == true.

// CHECK-LABEL: util.func public @branchDivergentForwardEdge
// CHECK-SAME: (%[[COND:.+]]: i1, %[[T0:.+]]: !stream.timepoint, %[[T1:.+]]: !stream.timepoint)
util.func public @branchDivergentForwardEdge(%cond: i1, %t0: !stream.timepoint, %t1: !stream.timepoint) -> !stream.timepoint {
  // CHECK: %[[T01:.+]] = stream.timepoint.join max(%[[T0]], %[[T1]])
  %t01 = stream.timepoint.join max(%t0, %t1) => !stream.timepoint
  // CHECK-NEXT: cf.cond_br
  // CHECK-SAME:   ^bb1(%[[T0]] : !stream.timepoint),
  // CHECK-SAME:   ^bb1(%[[T01]] : !stream.timepoint)
  cf.cond_br %cond, ^bb1(%t0 : !stream.timepoint), ^bb1(%t01 : !stream.timepoint)
// CHECK-NEXT: ^bb1(%[[BB1_ARG:.+]]: !stream.timepoint)
^bb1(%bb1_arg: !stream.timepoint):
  // CHECK: %[[T0_IMM:.+]] = stream.timepoint.immediate
  // CHECK-NEXT: %[[JOIN:.+]] = stream.timepoint.join max(%[[T0_IMM]], %[[T1]], %[[BB1_ARG]])
  %join = stream.timepoint.join max(%t0, %t1, %bb1_arg) => !stream.timepoint
  // CHECK: util.return %[[JOIN]]
  util.return %join : !stream.timepoint
}

// -----

// Tests that back edges with divergent timepoints don't get propagated.

// TODO(benvanik): some way of knowing %t0 is always covered; for now we aren't
// smart enough to track that through and likely need some
// must-be-executed-context-like machinery in order to do so. We just want to
// make sure we're preserving the timepoints here for correctness.

// CHECK-LABEL: util.func public @branchDivergentBackEdge
// CHECK-SAME: (%[[COND:.+]]: i1, %[[T0:.+]]: !stream.timepoint)
util.func public @branchDivergentBackEdge(%cond: i1, %t0: !stream.timepoint) -> !stream.timepoint {
  // CHECK: cf.br ^bb1
  cf.br ^bb1(%cond, %t0 : i1, !stream.timepoint)
// CHECK-NEXT: ^bb1(%[[BB1_COND:.+]]: i1, %[[BB1_T0:.+]]: !stream.timepoint)
^bb1(%bb1_cond: i1, %bb1_t0: !stream.timepoint):
  // CHECK-NOT: stream.timepoint.immediate
  // CHECK-NEXT: %[[BB1_T1:.+]] = stream.cmd.execute await(%[[BB1_T0]])
  %bb1_t1 = stream.cmd.execute await(%bb1_t0) => with() {} => !stream.timepoint
  // CHECK: %[[FALSE:.+]] = arith.constant false
  %cond_false = arith.constant false
  // CHECK-NEXT: cf.cond_br
  // CHECK-SAME:   ^bb1(%[[FALSE]], %[[BB1_T1]] : i1, !stream.timepoint)
  // CHECK-SAME:   ^bb2(%[[BB1_T1]] : !stream.timepoint)
  cf.cond_br %bb1_cond, ^bb1(%cond_false, %bb1_t1 : i1, !stream.timepoint), ^bb2(%bb1_t1 : !stream.timepoint)
// CHECK-NEXT: ^bb2(%[[BB2_T1:.+]]: !stream.timepoint)
^bb2(%bb2_t1: !stream.timepoint):
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[T0]], %[[BB2_T1]])
  %join = stream.timepoint.join max(%t0, %bb2_t1) => !stream.timepoint
  // CHECK-NEXT: util.return %[[JOIN]]
  util.return %join : !stream.timepoint
}

// -----

// Tests that scf.if regions with convergent yields are handled.
// Here %t0 is covered regardless of the %cond and can be elided.

// CHECK-LABEL: util.func public @scfIfConvergent
// CHECK-SAME: (%[[COND:.+]]: i1, %[[T0:.+]]: !stream.timepoint, %[[T1:.+]]: !stream.timepoint)
util.func public @scfIfConvergent(%cond: i1, %t0: !stream.timepoint, %t1: !stream.timepoint) -> !stream.timepoint {
  // CHECK: %[[IF:.+]] = scf.if
  %if = scf.if %cond -> !stream.timepoint {
    // CHECK: yield %[[T0]]
    scf.yield %t0 : !stream.timepoint
  } else {
    // CHECK: %[[T01:.+]] = stream.timepoint.join max(%[[T0]], %[[T1]])
    %t01 = stream.timepoint.join max(%t0, %t1) => !stream.timepoint
    // CHECK: yield %[[T01]]
    scf.yield %t01 : !stream.timepoint
  }
  // CHECK: %[[T0_IMM:.+]] = stream.timepoint.immediate
  // CHECK-NEXT: %[[JOIN:.+]] = stream.timepoint.join max(%[[T0_IMM]], %[[T1]], %[[IF]])
  %join = stream.timepoint.join max(%t0, %t1, %if) => !stream.timepoint
  // CHECK: util.return %[[JOIN]]
  util.return %join : !stream.timepoint
}

// TODO(benvanik): support scf.for

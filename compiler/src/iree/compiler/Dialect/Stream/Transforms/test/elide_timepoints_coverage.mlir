// RUN: iree-opt --split-input-file --iree-stream-elide-timepoints %s | FileCheck %s

// Tests that we don't (currently) do anything with global forwarding.
// Generic util analysis passes operating on globals can do things like folding.
// We just want to make sure here that we are preserving the global behavior.

util.global private mutable @global0 : !stream.timepoint
util.global private mutable @global1 : !stream.timepoint

util.initializer {
  %tp0 = stream.test.timeline_op with() : () -> () => !stream.timepoint
  util.global.store %tp0, @global0 : !stream.timepoint
  %tp1 = stream.test.timeline_op await(%tp0) => with() : () -> () => !stream.timepoint
  util.global.store %tp1, @global1 : !stream.timepoint
  util.return
}

// CHECK-LABEL: @initializedGlobals
util.func private @initializedGlobals() -> !stream.timepoint {
  // CHECK: %[[GLOBAL0:.+]] = util.global.load @global0
  %global0 = util.global.load @global0 : !stream.timepoint
  // CHECK: %[[GLOBAL1:.+]] = util.global.load @global1
  %global1 = util.global.load @global1 : !stream.timepoint
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[GLOBAL0]], %[[GLOBAL1]]) => !stream.timepoint
  %join = stream.timepoint.join max(%global0, %global1) => !stream.timepoint
  // CHECK: util.return %[[JOIN]]
  util.return %join : !stream.timepoint
}

// -----

// Tests that meaningful timeline ops are never marked immediate.

// CHECK-LABEL: @nonImmediate
util.func private @nonImmediate() -> !stream.timepoint {
  // CHECK: %[[TP:.+]] = stream.test.timeline_op with() : () -> () => !stream.timepoint
  %tp = stream.test.timeline_op with() : () -> () => !stream.timepoint
  // CHECK: util.return %[[TP]]
  util.return %tp : !stream.timepoint
}

// -----

// Tests that coverage propagates through timeline ops. Here %tp0 is covered
// by both %tp1a and %tp1b and does not need to be joined.

// CHECK-LABEL: @joinChained
util.func public @joinChained() -> !stream.timepoint {
  // CHECK: %[[TP0:.+]] = stream.test.timeline_op with() : () -> () => !stream.timepoint
  %tp0 = stream.test.timeline_op with() : () -> () => !stream.timepoint
  // CHECK: %[[TP1A:.+]] = stream.test.timeline_op await(%[[TP0]]) => with() : () -> () => !stream.timepoint
  %tp1a = stream.test.timeline_op await(%tp0) => with() : () -> () => !stream.timepoint
  // CHECK: %[[TP1B:.+]] = stream.test.timeline_op await(%[[TP0]]) => with() : () -> () => !stream.timepoint
  %tp1b = stream.test.timeline_op await(%tp0) => with() : () -> () => !stream.timepoint
  // CHECK: %[[TP0_IMM:.+]] = stream.timepoint.immediate
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[TP0_IMM]], %[[TP1A]], %[[TP1B]]) => !stream.timepoint
  %join = stream.timepoint.join max(%tp0, %tp1a, %tp1b) => !stream.timepoint
  // CHECK: util.return %[[JOIN]]
  util.return %join : !stream.timepoint
}

// -----

// Tests that coverage propagates through a select: %tp0 is covered by both
// the true and false conditions and does not need to be joined.

// CHECK-LABEL: @selectCovered
// CHECK-SAME: (%[[COND:.+]]: i1)
util.func public @selectCovered(%cond: i1) -> !stream.timepoint {
  // CHECK: %[[TP0:.+]] = stream.test.timeline_op with() : () -> () => !stream.timepoint
  %tp0 = stream.test.timeline_op with() : () -> () => !stream.timepoint
  // CHECK: %[[TP1A:.+]] = stream.test.timeline_op await(%[[TP0]]) => with() : () -> () => !stream.timepoint
  %tp1a = stream.test.timeline_op await(%tp0) => with() : () -> () => !stream.timepoint
  // CHECK: %[[TP1B:.+]] = stream.test.timeline_op await(%[[TP0]]) => with() : () -> () => !stream.timepoint
  %tp1b = stream.test.timeline_op await(%tp0) => with() : () -> () => !stream.timepoint
  // CHECK: %[[SELECT:.+]] = arith.select %[[COND]], %[[TP1A]], %[[TP1B]] : !stream.timepoint
  %select = arith.select %cond, %tp1a, %tp1b : !stream.timepoint
  // CHECK: %[[TP0_IMM:.+]] = stream.timepoint.immediate
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[TP0_IMM]], %[[SELECT]]) => !stream.timepoint
  %join = stream.timepoint.join max(%tp0, %select) => !stream.timepoint
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
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[T0_COVERED]], %[[T1_COVERED]], %[[CALL]]) => !stream.timepoint
  %join = stream.timepoint.join max(%t0, %t1, %call) => !stream.timepoint
  // CHECK: util.return %[[JOIN]]
  util.return %join : !stream.timepoint
}
// CHECK-LABEL: util.func private @callee
util.func private @callee(%t0a: !stream.timepoint, %t0b: !stream.timepoint) -> !stream.timepoint {
  // CHECK-NOT: stream.timepoint.immediate
  // CHECK: %[[T1:.+]] = stream.timepoint.join max
  %t1 = stream.timepoint.join max(%t0a, %t0b) => !stream.timepoint
  // CHECK: util.return %[[T1]]
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
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[UNKNOWN_IMM]], %[[CALL]]#0, %[[CALL]]#1) => !stream.timepoint
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
  // CHECK: %[[T01:.+]] = stream.timepoint.join max(%[[T0]], %[[T1]]) => !stream.timepoint
  %t01 = stream.timepoint.join max(%t0, %t1) => !stream.timepoint
  // CHECK: %[[CALL1:.+]] = util.call @nonUniformCallee(%[[T01]])
  %call1 = util.call @nonUniformCallee(%t01) : (!stream.timepoint) -> !stream.timepoint
  // CHECK: %[[T0_IMM:.+]] = stream.timepoint.immediate
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[T0_IMM]], %[[CALL0]], %[[T1]], %[[CALL1]]) => !stream.timepoint
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
  // CHECK-NEXT: %[[JOIN:.+]] = stream.timepoint.join max(%[[T0_IMM]], %[[BB1_T0]]) => !stream.timepoint
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
  // CHECK: %[[T1A:.+]] = stream.test.timeline_op await(%[[T0]]) => with() : () -> () => !stream.timepoint
  %t1a = stream.test.timeline_op await(%t0) => with() : () -> () => !stream.timepoint
  // CHECK: %[[T1B:.+]] = stream.test.timeline_op await(%[[T0]]) => with() : () -> () => !stream.timepoint
  %t1b = stream.test.timeline_op await(%t0) => with() : () -> () => !stream.timepoint
  // CHECK: cf.cond_br %[[COND]]
  // CHECK-SAME:   ^bb1(%[[T1A]] : !stream.timepoint),
  // CHECK-SAME:   ^bb1(%[[T1B]] : !stream.timepoint)
  cf.cond_br %cond, ^bb1(%t1a : !stream.timepoint), ^bb1(%t1b : !stream.timepoint)
// CHECK-NEXT: ^bb1(%[[BB1_ARG:.+]]: !stream.timepoint)
^bb1(%bb1_arg: !stream.timepoint):
  // CHECK: %[[T0_IMM:.+]] = stream.timepoint.immediate
  // CHECK-NEXT: %[[JOIN:.+]] = stream.timepoint.join max(%[[T0_IMM]], %[[BB1_ARG]]) => !stream.timepoint
  %join = stream.timepoint.join max(%t0, %bb1_arg) => !stream.timepoint
  // CHECK: util.return %[[JOIN]]
  util.return %join : !stream.timepoint
}

// -----

// Tests that forward edges with divergent timepoint coverage get propagated.
// %t0 is covered on both paths but %t1 is only covered when %cond == true.

// CHECK-LABEL: util.func public @branchDivergentForwardEdge
// CHECK-SAME: (%{{.+}}: i1, %[[T0:.+]]: !stream.timepoint, %[[T1:.+]]: !stream.timepoint)
util.func public @branchDivergentForwardEdge(%cond: i1, %t0: !stream.timepoint, %t1: !stream.timepoint) -> !stream.timepoint {
  // CHECK: %[[T01:.+]] = stream.timepoint.join max(%[[T0]], %[[T1]]) => !stream.timepoint
  %t01 = stream.timepoint.join max(%t0, %t1) => !stream.timepoint
  // CHECK-NEXT: cf.cond_br
  // CHECK-SAME:   ^bb1(%[[T0]] : !stream.timepoint),
  // CHECK-SAME:   ^bb1(%[[T01]] : !stream.timepoint)
  cf.cond_br %cond, ^bb1(%t0 : !stream.timepoint), ^bb1(%t01 : !stream.timepoint)
// CHECK-NEXT: ^bb1(%[[BB1_ARG:.+]]: !stream.timepoint)
^bb1(%bb1_arg: !stream.timepoint):
  // CHECK: %[[T0_IMM:.+]] = stream.timepoint.immediate
  // CHECK-NEXT: %[[JOIN:.+]] = stream.timepoint.join max(%[[T0_IMM]], %[[T1]], %[[BB1_ARG]]) => !stream.timepoint
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
// CHECK-SAME: (%{{.+}}: i1, %[[T0:.+]]: !stream.timepoint)
util.func public @branchDivergentBackEdge(%cond: i1, %t0: !stream.timepoint) -> !stream.timepoint {
  // CHECK: cf.br ^bb1
  cf.br ^bb1(%cond, %t0 : i1, !stream.timepoint)
// CHECK-NEXT: ^bb1(%[[BB1_COND:.+]]: i1, %[[BB1_T0:.+]]: !stream.timepoint)
^bb1(%bb1_cond: i1, %bb1_t0: !stream.timepoint):
  // CHECK-NOT: stream.timepoint.immediate
  // CHECK-NEXT: %[[BB1_T1:.+]] = stream.test.timeline_op await(%[[BB1_T0]]) => with() : () -> () => !stream.timepoint
  %bb1_t1 = stream.test.timeline_op await(%bb1_t0) => with() : () -> () => !stream.timepoint
  // CHECK: %[[FALSE:.+]] = arith.constant false
  %cond_false = arith.constant false
  // CHECK-NEXT: cf.cond_br
  // CHECK-SAME:   ^bb1(%[[FALSE]], %[[BB1_T1]] : i1, !stream.timepoint)
  // CHECK-SAME:   ^bb2(%[[BB1_T1]] : !stream.timepoint)
  cf.cond_br %bb1_cond, ^bb1(%cond_false, %bb1_t1 : i1, !stream.timepoint), ^bb2(%bb1_t1 : !stream.timepoint)
// CHECK-NEXT: ^bb2(%[[BB2_T1:.+]]: !stream.timepoint)
^bb2(%bb2_t1: !stream.timepoint):
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[T0]], %[[BB2_T1]]) => !stream.timepoint
  %join = stream.timepoint.join max(%t0, %bb2_t1) => !stream.timepoint
  // CHECK-NEXT: util.return %[[JOIN]]
  util.return %join : !stream.timepoint
}

// -----

// Tests that scf.if regions with convergent yields are handled.
// Here %t0 is covered regardless of the %cond and can be elided.

// CHECK-LABEL: util.func public @scfIfConvergent
// CHECK-SAME: (%{{.+}}: i1, %[[T0:.+]]: !stream.timepoint, %[[T1:.+]]: !stream.timepoint)
util.func public @scfIfConvergent(%cond: i1, %t0: !stream.timepoint, %t1: !stream.timepoint) -> !stream.timepoint {
  // CHECK: %[[IF:.+]] = scf.if
  %if = scf.if %cond -> !stream.timepoint {
    // CHECK: yield %[[T0]]
    scf.yield %t0 : !stream.timepoint
  } else {
    // CHECK: %[[T01:.+]] = stream.timepoint.join max(%[[T0]], %[[T1]]) => !stream.timepoint
    %t01 = stream.timepoint.join max(%t0, %t1) => !stream.timepoint
    // CHECK: yield %[[T01]]
    scf.yield %t01 : !stream.timepoint
  }
  // CHECK: %[[T0_IMM:.+]] = stream.timepoint.immediate
  // CHECK-NEXT: %[[JOIN:.+]] = stream.timepoint.join max(%[[T0_IMM]], %[[T1]], %[[IF]]) => !stream.timepoint
  %join = stream.timepoint.join max(%t0, %t1, %if) => !stream.timepoint
  // CHECK: util.return %[[JOIN]]
  util.return %join : !stream.timepoint
}

// TODO(benvanik): support scf.for

// -----

// Tests that timeline-aware ops propagate coverage.
// The timepoint imported from the wait fence is covered by the result timepoint
// from the signal fence.

// CHECK-LABEL: util.func public @timelineAwareCallCoverage
// CHECK-SAME: (%[[VIEW:.+]]: !hal.buffer_view, %[[SIGNAL_FENCE:.+]]: !stream.test.fence)
util.func public @timelineAwareCallCoverage(%view: !hal.buffer_view, %signal_fence: !stream.test.fence) -> !hal.buffer_view {
  // Create async work producing a timepoint.
  // CHECK: %[[T0:.+]] = stream.test.timeline_op with() : () -> () => !stream.timepoint
  %t0 = stream.test.timeline_op with() : () -> () => !stream.timepoint

  // Export timepoint as fence for the call.
  %wait_fence = stream.timepoint.export %t0 => (!stream.test.fence)

  // Timeline-aware call - timepoints built from fences.
  // CHECK: stream.test.timeline_aware(%[[VIEW]]) waits(%{{.+}}) signals(%[[SIGNAL_FENCE]])
  %result = stream.test.timeline_aware(%view) waits(%wait_fence) signals(%signal_fence) : (!hal.buffer_view) -> !hal.buffer_view

  // Import signal fence as timepoint.
  // CHECK: %[[T1:.+]] = stream.timepoint.import %[[SIGNAL_FENCE]] : (!stream.test.fence)
  %t1 = stream.timepoint.import %signal_fence : (!stream.test.fence) => !stream.timepoint

  // CHECK: %[[T0_IMM:.+]] = stream.timepoint.immediate
  // CHECK: %{{.+}} = stream.timepoint.join max(%[[T0_IMM]], %[[T1]]) => !stream.timepoint
  %join = stream.timepoint.join max(%t0, %t1) => !stream.timepoint

  // CHECK: util.return
  util.return %result : !hal.buffer_view
}

// -----

// Tests that timeline-aware calls in sequence propagate coverage correctly.
// Each call's signal fence covers its wait fence.

// CHECK-LABEL: util.func public @timelineAwareCallSequence
// CHECK-SAME: (%[[VIEW:.+]]: !hal.buffer_view, %[[FENCE_A_SIGNAL:.+]]: !stream.test.fence, %[[FENCE_B_SIGNAL:.+]]: !stream.test.fence)
util.func public @timelineAwareCallSequence(%view: !hal.buffer_view, %fence_a_signal: !stream.test.fence, %fence_b_signal: !stream.test.fence) -> !hal.buffer_view {
  // First async work.
  // CHECK: %[[T0:.+]] = stream.test.timeline_op with() : () -> () => !stream.timepoint
  %t0 = stream.test.timeline_op with() : () -> () => !stream.timepoint

  // First timeline-aware call.
  %fence_a_wait = stream.timepoint.export %t0 => (!stream.test.fence)
  // CHECK: stream.test.timeline_aware(%[[VIEW]]) waits(%{{.+}}) signals(%[[FENCE_A_SIGNAL]])
  %result_a = stream.test.timeline_aware(%view) waits(%fence_a_wait) signals(%fence_a_signal) : (!hal.buffer_view) -> !hal.buffer_view
  // CHECK: %[[T1:.+]] = stream.timepoint.import %[[FENCE_A_SIGNAL]] : (!stream.test.fence)
  %t1 = stream.timepoint.import %fence_a_signal : (!stream.test.fence) => !stream.timepoint

  // Second async work awaiting first call.
  // CHECK: %[[T2:.+]] = stream.test.timeline_op await(%[[T1]]) => with() : () -> () => !stream.timepoint
  %t2 = stream.test.timeline_op await(%t1) => with() : () -> () => !stream.timepoint

  // Second timeline-aware call.
  %fence_b_wait = stream.timepoint.export %t2 => (!stream.test.fence)
  // CHECK: stream.test.timeline_aware(%{{.+}}) waits(%{{.+}}) signals(%[[FENCE_B_SIGNAL]])
  %result_b = stream.test.timeline_aware(%result_a) waits(%fence_b_wait) signals(%fence_b_signal) : (!hal.buffer_view) -> !hal.buffer_view
  // CHECK: %[[T3:.+]] = stream.timepoint.import %[[FENCE_B_SIGNAL]] : (!stream.test.fence)
  %t3 = stream.timepoint.import %fence_b_signal : (!stream.test.fence) => !stream.timepoint

  // All of %t0, %t1, and %t2 should be elided since they're covered by %t3.
  // %t3 transitively covers: %t2 (direct), %t1 (via %t2's await), %t0 (via %t1).
  // CHECK-DAG: %[[T0_IMM:.+]] = stream.timepoint.immediate
  // CHECK-DAG: %[[T1_IMM:.+]] = stream.timepoint.immediate
  // CHECK-DAG: %[[T2_IMM:.+]] = stream.timepoint.immediate
  // CHECK: %{{.+}} = stream.timepoint.join max(%[[T0_IMM]], %[[T1_IMM]], %[[T2_IMM]], %[[T3]]) => !stream.timepoint
  %join = stream.timepoint.join max(%t0, %t1, %t2, %t3) => !stream.timepoint

  // CHECK: util.return
  util.return %result_b : !hal.buffer_view
}

// -----

// Tests that non-timeline-aware calls (without fences) are not treated specially.

util.func private @normal_func(!hal.buffer_view) -> !hal.buffer_view

// CHECK-LABEL: util.func public @nonTimelineAwareCall
util.func public @nonTimelineAwareCall(%view: !hal.buffer_view) -> !hal.buffer_view {
  // CHECK: %[[T0:.+]] = stream.test.timeline_op with() : () -> () => !stream.timepoint
  %t0 = stream.test.timeline_op with() : () -> () => !stream.timepoint

  // Normal call without fences - not timeline-aware.
  // CHECK: util.call @normal_func
  %result = util.call @normal_func(%view) : (!hal.buffer_view) -> !hal.buffer_view

  // Create another timepoint after the call.
  // CHECK: %[[T1:.+]] = stream.test.timeline_op with() : () -> () => !stream.timepoint
  %t1 = stream.test.timeline_op with() : () -> () => !stream.timepoint

  // Both timepoints should remain since there's no coverage relationship.
  // CHECK-NOT: stream.timepoint.immediate
  // CHECK: %{{.+}} = stream.timepoint.join max(%[[T0]], %[[T1]]) => !stream.timepoint
  %join = stream.timepoint.join max(%t0, %t1) => !stream.timepoint

  // CHECK: util.return
  util.return %result : !hal.buffer_view
}

// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-stream-elide-timepoints)" %s | FileCheck %s

// Tests proactive await folding: absorbing awaits into timeline ops that don't
// already have them, and reactive cleanup: eliminating redundant awaits when
// timeline ops already cover them.

// Tests that await is absorbed into stream.test.timeline_op when it has no await
// clause.

// CHECK-LABEL: @awaitFoldIntoAsyncExecute
util.func public @awaitFoldIntoAsyncExecute(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c8 = arith.constant 8 : index

  // Produce async value + timepoint.
  // CHECK: %[[CLONE:.+]], %[[TP:.+]] = stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await should be eliminated.
  // CHECK-NOT: stream.timepoint.await
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}

  // Execute should absorb the await.
  // CHECK: %[[RESULT:.+]], %{{.+}} = stream.test.timeline_op await(%[[TP]]) => with(%[[CLONE]])
  %result, %result_tp = stream.test.timeline_op with(%awaited) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<external>
}

// -----

// Tests that await is absorbed into stream.cmd.execute when it has no await
// clause.

// CHECK-LABEL: @awaitFoldIntoCmdExecute
util.func public @awaitFoldIntoCmdExecute(%arg0: !stream.resource<external>) -> !stream.timepoint {
  %c8 = arith.constant 8 : index

  // Produce async value + timepoint.
  // CHECK: %[[CLONE:.+]], %[[TP:.+]] = stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await should be eliminated.
  // CHECK-NOT: stream.timepoint.await
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}

  // Cmd.execute should absorb the await.
  // CHECK: %[[RESULT_TP:.+]] = stream.cmd.execute await(%[[TP]]) =>
  // CHECK-SAME: with(%[[CLONE]] as %{{.+}}: !stream.resource<external>{%c8})
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %result_tp = stream.cmd.execute with(%awaited as %a: !stream.resource<external>{%c8}) {
    stream.cmd.concurrent {
      stream.cmd.fill %c0_i32, %a[%c0 for %c8] : i32 -> !stream.resource<external>{%c8}
    }
  } => !stream.timepoint

  // CHECK: util.return %[[RESULT_TP]]
  util.return %result_tp : !stream.timepoint
}

// -----

// Tests that await is absorbed into timepoint.join when it has no await clause.

// CHECK-LABEL: @awaitFoldIntoTimepointJoin
util.func public @awaitFoldIntoTimepointJoin(%arg0: !stream.resource<external>) -> !stream.timepoint {
  %c8 = arith.constant 8 : index

  // Produce async value + timepoint.
  // CHECK: stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await should be eliminated.
  // CHECK-NOT: stream.timepoint.await
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}

  // Create another independent operation.
  // CHECK: %[[CLONE2:.+]], %[[TP2:.+]] = stream.test.timeline_op
  %clone2, %tp2 = stream.test.timeline_op with(%awaited) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Join only needs %tp2 since %tp was absorbed into the second execute.
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[TP2]])
  %join = stream.timepoint.join max(%tp2) => !stream.timepoint

  // CHECK: util.return %[[JOIN]]
  util.return %join : !stream.timepoint
}

// -----

// Tests that await is merged with an existing await by creating a join.

// CHECK-LABEL: @awaitFoldWithExistingAwait
util.func public @awaitFoldWithExistingAwait(%arg0: !stream.resource<external>, %arg1: !stream.resource<external>) -> !stream.resource<external> {
  %c8 = arith.constant 8 : index

  // First async operation.
  // CHECK: %[[CLONE1:.+]], %[[TP1:.+]] = stream.test.timeline_op
  %clone1, %tp1 = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Second async operation.
  // CHECK: %[[CLONE2:.+]], %[[TP2:.+]] = stream.test.timeline_op
  %clone2, %tp2 = stream.test.timeline_op with(%arg1) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await first operation.
  // CHECK-NOT: stream.timepoint.await %[[TP1]]
  %awaited1 = stream.timepoint.await %tp1 => %clone1 : !stream.resource<external>{%c8}

  // Execute already has await on tp2, should create join with tp1.
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[TP2]], %[[TP1]])
  // CHECK: %[[RESULT:.+]], %{{.+}} = stream.test.timeline_op await_limit(1) await(%[[JOIN]]) =>
  // CHECK-SAME: with(%[[CLONE1]], %[[CLONE2]])
  %result, %result_tp = stream.test.timeline_op await_limit(1) await(%tp2) => with(%awaited1, %clone2) : (!stream.resource<external>{%c8}, !stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<external>
}

// -----

// Tests that await is NOT folded when async value doesn't dominate the
// consumer.

// CHECK-LABEL: @awaitNoFoldAcrossRegionBoundary
util.func public @awaitNoFoldAcrossRegionBoundary(%cond: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0_i32 = arith.constant 0 : i32
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // scf.if produces a value defined inside a branch.
  // CHECK: %[[IF_RESULT:.+]] = scf.if
  %result = scf.if %cond -> !stream.resource<external> {
    // Value defined inside then branch.
    %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

    // Await inside branch.
    %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}
    scf.yield %awaited : !stream.resource<external>
  } else {
    %splat = stream.async.splat %c0_i32 : i32 -> !stream.resource<external>{%c4}
    scf.yield %splat : !stream.resource<external>
  }

  // Consumer outside the branch - should NOT fold because %clone doesn't
  // dominate.
  // The final execute should not have an await clause.
  // CHECK: %[[FINAL:.+]], %{{.+}} = stream.test.timeline_op with(%[[IF_RESULT]])
  %final, %final_tp = stream.test.timeline_op with(%result) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // CHECK: util.return %[[FINAL]]
  util.return %final : !stream.resource<external>
}

// -----

// Tests reactive cleanup: await is eliminated when already covered.

// CHECK-LABEL: @awaitEliminateWhenCovered
util.func public @awaitEliminateWhenCovered(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c8 = arith.constant 8 : index

  // Produce async value + timepoint.
  // CHECK: %[[CLONE:.+]], %[[TP:.+]] = stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await should be eliminated (reactive cleanup).
  // CHECK-NOT: stream.timepoint.await
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}

  // Execute ALREADY has await(%tp), so awaited result is redundant.
  // Should use %clone directly instead of %awaited.
  // CHECK: %[[RESULT:.+]], %{{.+}} = stream.test.timeline_op await(%[[TP]]) =>
  // CHECK-SAME: with(%[[CLONE]])
  %result, %result_tp = stream.test.timeline_op await(%tp) => with(%awaited) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<external>
}

// -----

// Tests transitive coverage: tp2 covers tp1, so await on tp1 can be eliminated.

// CHECK-LABEL: @awaitEliminateWhenCoveredTransitive
util.func public @awaitEliminateWhenCoveredTransitive(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c8 = arith.constant 8 : index

  // First async operation produces tp1.
  // CHECK: %[[CLONE1:.+]], %[[TP1:.+]] = stream.test.timeline_op
  %clone1, %tp1 = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // First await on tp1.
  // CHECK-NOT: stream.timepoint.await %[[TP1]]
  %awaited1 = stream.timepoint.await %tp1 => %clone1 : !stream.resource<external>{%c8}

  // Second async operation produces tp2 (which covers tp1).
  // CHECK: %[[CLONE2:.+]], %[[TP2:.+]] = stream.test.timeline_op await(%[[TP1]]) =>
  %clone2, %tp2 = stream.test.timeline_op await(%tp1) => with(%awaited1) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Second await on tp2 (which already covers tp1).
  // Both awaited values are eliminated since they'll be folded into execute.
  // CHECK-NOT: stream.timepoint.await
  %awaited2 = stream.timepoint.await %tp2 => %clone2 : !stream.resource<external>{%c8}

  // Execute uses both awaited1 and awaited2, creating joins to merge timepoints.
  // Joins are created because both %clone1 and %clone2 need to be used.
  // CHECK-DAG: %[[JOIN1:.+]] = stream.timepoint.join max(%[[TP2]], %[[TP1]])
  // CHECK-DAG: %[[JOIN2:.+]] = stream.timepoint.join max(%[[JOIN1]], %[[TP2]])
  // CHECK: %[[RESULT:.+]], %{{.+}} = stream.test.timeline_op await_limit(1) await(%[[JOIN2]]) =>
  // CHECK-SAME: with(%[[CLONE1]], %[[CLONE2]])
  %result, %result_tp = stream.test.timeline_op await_limit(1) await(%tp2) => with(%awaited1, %awaited2) : (!stream.resource<external>{%c8}, !stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<external>
}

// -----

// Tests that await is NOT folded when timeline op's await doesn't cover it.

// CHECK-LABEL: @awaitNoFoldWhenNotCovered
util.func public @awaitNoFoldWhenNotCovered(%arg0: !stream.resource<external>, %arg1: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // First async operation produces tp1.
  // CHECK: %[[CLONE1:.+]], %[[TP1:.+]] = stream.test.timeline_op
  %clone1, %tp1 = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Second async operation produces tp2 (independent, doesn't cover tp1).
  // CHECK: %[[CLONE2:.+]], %[[TP2:.+]] = stream.test.timeline_op
  %clone2, %tp2 = stream.test.timeline_op with(%arg1) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await on tp1.
  // Await is eliminated and a join is created instead.
  // CHECK-NOT: stream.timepoint.await %[[TP1]]
  %awaited1 = stream.timepoint.await %tp1 => %clone1 : !stream.resource<external>{%c8}

  // Execute awaits tp2, which does NOT cover tp1.
  // Pass creates a join to wait for both timepoints.
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[TP2]], %[[TP1]])
  // CHECK: %[[RESULT:.+]], %{{.+}} = stream.test.timeline_op await_limit(1) await(%[[JOIN]]) =>
  // CHECK-SAME: with(%[[CLONE1]], %[[CLONE2]])
  %result, %result_tp = stream.test.timeline_op await_limit(1) await(%tp2) => with(%awaited1, %clone2) : (!stream.resource<external>{%c8}, !stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<external>
}

// -----

// Tests that await folding works across call boundaries.

// CHECK-LABEL: util.func private @awaitFoldAcrossCallBoundary_callee
util.func private @awaitFoldAcrossCallBoundary_callee(%arg: !stream.resource<external>) -> !stream.resource<external> {
  %c8 = arith.constant 8 : index

  // Callee uses the value in a timeline op.
  // CHECK: %[[RESULT:.+]], %{{.+}} = stream.test.timeline_op
  // CHECK-SAME: with(%arg0)
  %result, %result_tp = stream.test.timeline_op with(%arg) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<external>
}

// CHECK-LABEL: @awaitFoldAcrossCallBoundary
util.func public @awaitFoldAcrossCallBoundary(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c8 = arith.constant 8 : index

  // Produce async value + timepoint.
  // CHECK: stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await in caller.
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}

  // Pass awaited value to callee.
  // walkTransitiveUses should follow call and find timeline op in callee.
  %result = util.call @awaitFoldAcrossCallBoundary_callee(%awaited) : (!stream.resource<external>) -> !stream.resource<external>

  util.return %result : !stream.resource<external>
}

// -----

// Tests that await folding works across global store/load.

// CHECK-LABEL: util.global private mutable @awaitFoldGlobal

util.global private mutable @awaitFoldGlobal : !stream.resource<external>

// CHECK-LABEL: @awaitFoldAcrossGlobalStoreLoad_store
util.func public @awaitFoldAcrossGlobalStoreLoad_store(%arg0: !stream.resource<external>) {
  %c8 = arith.constant 8 : index

  // Produce async value + timepoint.
  // CHECK: stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await and store to global.
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}
  util.global.store %awaited, @awaitFoldGlobal : !stream.resource<external>

  util.return
}

// CHECK-LABEL: @awaitFoldAcrossGlobalStoreLoad_load
util.func public @awaitFoldAcrossGlobalStoreLoad_load() -> !stream.resource<external> {
  %c8 = arith.constant 8 : index

  // Load from global.
  %loaded = util.global.load @awaitFoldGlobal : !stream.resource<external>

  // Use in timeline op - walkTransitiveUses should trace back through global.
  // CHECK: %[[RESULT:.+]], %{{.+}} = stream.test.timeline_op
  %result, %result_tp = stream.test.timeline_op with(%loaded) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<external>
}

// -----

// Tests dominance: value from outer scope used in nested scf.if (should work).

// CHECK-LABEL: @awaitFoldOuterToInnerScope
util.func public @awaitFoldOuterToInnerScope(%outer_cond: i1, %inner_cond: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0_i32 = arith.constant 0 : i32
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // Value defined in outer scope.
  // CHECK: %[[CLONE:.+]], %[[TP:.+]] = stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await in outer scope.
  // CHECK-NOT: stream.timepoint.await
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}

  // Outer scf.if.
  // CHECK: %{{.+}} = scf.if
  %result = scf.if %outer_cond -> !stream.resource<external> {
    // Inner scf.if uses awaited value from outer scope.
    // CHECK: %{{.+}} = scf.if
    %inner_result = scf.if %inner_cond -> !stream.resource<external> {
      // Should fold - %clone dominates this use.
      // CHECK: %[[EXEC:.+]], %{{.+}} = stream.test.timeline_op await(%[[TP]]) =>
      // CHECK-SAME: with(%[[CLONE]])
      %exec, %exec_tp = stream.test.timeline_op with(%awaited) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint
      // CHECK: scf.yield %[[EXEC]]
      scf.yield %exec : !stream.resource<external>
    } else {
      %splat = stream.async.splat %c0_i32 : i32 -> !stream.resource<external>{%c4}
      scf.yield %splat : !stream.resource<external>
    }
    scf.yield %inner_result : !stream.resource<external>
  } else {
    %splat = stream.async.splat %c0_i32 : i32 -> !stream.resource<external>{%c4}
    scf.yield %splat : !stream.resource<external>
  }

  util.return %result : !stream.resource<external>
}

// -----

// Tests scf.for with await value used across iterations.

// CHECK-LABEL: @awaitWithScfForLoop
util.func public @awaitWithScfForLoop(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c8 = arith.constant 8 : index

  // Produce async value + timepoint.
  // CHECK: stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await before loop.
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}

  // Loop uses awaited value in each iteration.
  // CHECK: scf.for
  %result = scf.for %i = %c0 to %c3 step %c1 iter_args(%iter = %awaited) -> (!stream.resource<external>) {
    // Each iteration uses iter value in timeline op.
    // CHECK: stream.test.timeline_op
    %cloned, %cloned_tp = stream.test.timeline_op with(%iter) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint
    scf.yield %cloned : !stream.resource<external>
  }

  util.return %result : !stream.resource<external>
}

// -----

// Tests await used in BOTH timeline op and non-timeline op (condition).

// CHECK-LABEL: @awaitMixedTimelineAndNonTimelineUse
util.func public @awaitMixedTimelineAndNonTimelineUse(%arg0: !stream.resource<external>, %arg1: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // Produce async value + timepoint.
  // CHECK: %[[CLONE:.+]], %[[TP:.+]] = stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await should be eliminated from timeline op use, but preserved for comparison.
  // CHECK: stream.timepoint.await %[[TP]] => %[[CLONE]]
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}

  // Non-timeline use: pointer comparison.
  %cond = util.cmp.eq %awaited, %arg1 : !stream.resource<external>

  // Timeline use in scf.if.
  // CHECK: scf.if
  %result = scf.if %cond -> !stream.resource<external> {
    // Should fold for timeline op use.
    // CHECK: %[[EXEC:.+]], %{{.+}} = stream.test.timeline_op await(%[[TP]]) =>
    // CHECK-SAME: with(%[[CLONE]])
    %exec, %exec_tp = stream.test.timeline_op with(%awaited) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint
    // CHECK: scf.yield %[[EXEC]]
    scf.yield %exec : !stream.resource<external>
  } else {
    scf.yield %arg1 : !stream.resource<external>
  }

  util.return %result : !stream.resource<external>
}

// -----

// Tests nested stream.test.timeline_op regions.

// CHECK-LABEL: @awaitWithNestedAsyncExecute
util.func public @awaitWithNestedAsyncExecute(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c8 = arith.constant 8 : index

  // Outer stream.test.timeline_op.
  // CHECK: %[[OUTER:.+]], %[[OUTER_TP:.+]] = stream.test.timeline_op
  %outer, %outer_tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await outer result.
  // CHECK-NOT: stream.timepoint.await %[[OUTER_TP]]
  %awaited_outer = stream.timepoint.await %outer_tp => %outer : !stream.resource<external>{%c8}

  // Inner stream.test.timeline_op that uses awaited outer result.
  // CHECK: %[[INNER:.+]], %{{.+}} = stream.test.timeline_op await(%[[OUTER_TP]]) =>
  // CHECK-SAME: with(%[[OUTER]])
  %inner, %inner_tp = stream.test.timeline_op with(%awaited_outer) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // CHECK: util.return %[[INNER]]
  util.return %inner : !stream.resource<external>
}

// -----

// Tests scf.for result used outside loop (block argument vs result dominance).

// CHECK-LABEL: @awaitScfForResultDominance
util.func public @awaitScfForResultDominance(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c8 = arith.constant 8 : index

  // Initial value.
  %init, %init_tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Loop produces new value each iteration.
  // CHECK: %[[LOOP_RESULT:.+]] = scf.for
  %loop_result = scf.for %i = %c0 to %c3 step %c1 iter_args(%iter = %init) -> (!stream.resource<external>) {
    // Value produced inside loop.
    %cloned, %cloned_tp = stream.test.timeline_op with(%iter) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint
    %awaited = stream.timepoint.await %cloned_tp => %cloned : !stream.resource<external>{%c8}
    scf.yield %awaited : !stream.resource<external>
  }

  // Use loop result outside loop - should work, loop result dominates.
  // CHECK: %[[FINAL:.+]], %{{.+}} = stream.test.timeline_op with(%[[LOOP_RESULT]])
  // CHECK-NEXT: util.return %[[FINAL]]
  %final, %final_tp = stream.test.timeline_op with(%loop_result) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  util.return %final : !stream.resource<external>
}

// -----

// Tests complex coverage chain with multiple timepoint joins.

// CHECK-LABEL: @awaitComplexCoverageChain
util.func public @awaitComplexCoverageChain(%arg0: !stream.resource<external>, %arg1: !stream.resource<external>, %arg2: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // Three independent operations.
  // CHECK: %[[CLONE1:.+]], %[[TP1:.+]] = stream.test.timeline_op
  %clone1, %tp1 = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // CHECK: %[[CLONE2:.+]], %[[TP2:.+]] = stream.test.timeline_op
  %clone2, %tp2 = stream.test.timeline_op with(%arg1) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // CHECK: %[[CLONE3:.+]], %[[TP3:.+]] = stream.test.timeline_op
  %clone3, %tp3 = stream.test.timeline_op with(%arg2) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Join tp1 and tp2.
  // CHECK: %[[JOIN12:.+]] = stream.timepoint.join max(%[[TP1]], %[[TP2]])
  %join12 = stream.timepoint.join max(%tp1, %tp2) => !stream.timepoint

  // Await tp1 and tp3.
  %awaited1 = stream.timepoint.await %tp1 => %clone1 : !stream.resource<external>{%c8}
  %awaited3 = stream.timepoint.await %tp3 => %clone3 : !stream.resource<external>{%c8}

  // Operation awaiting join12 using awaited1 and awaited3.
  // Pass absorbs all timepoints directly (no await_limit, so no additional joins).
  // CHECK-NOT: stream.timepoint.await
  // CHECK: %[[RESULT:.+]], %{{.+}} = stream.test.timeline_op await(%[[JOIN12]], %[[TP1]], %[[TP3]]) =>
  // CHECK-SAME: with(%[[CLONE1]], %[[CLONE3]])
  %result, %result_tp = stream.test.timeline_op await(%join12) => with(%awaited1, %awaited3) : (!stream.resource<external>{%c8}, !stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<external>
}

// -----

// Tests that await folding in CF loops correctly handles block arguments.
// The async value must NOT be used to replace block arguments when it doesn't
// dominate them (block arguments are defined at block entry).

// CHECK-LABEL: @awaitFoldCFLoopBlockArgument
// CHECK-SAME: (%[[COND:.+]]: i1)
util.func public @awaitFoldCFLoopBlockArgument(%cond: i1) -> !stream.resource<external> {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32

  // Initial resource.
  // CHECK: %[[INIT:.+]], %[[INIT_TP:.+]] = stream.test.timeline_op
  %init, %init_tp = stream.test.timeline_op with() : () -> !stream.resource<external>{%c4} => !stream.timepoint

  // CHECK: %[[INIT_SYNC:.+]] = stream.timepoint.await %[[INIT_TP]] => %[[INIT]]
  %init_sync = stream.timepoint.await %init_tp => %init : !stream.resource<external>{%c4}

  // CHECK: cf.cond_br %[[COND]], ^bb1(%[[INIT_SYNC]] : !stream.resource<external>), ^bb2(%[[INIT_SYNC]] : !stream.resource<external>)
  cf.cond_br %cond, ^bb1(%init_sync : !stream.resource<external>), ^bb2(%init_sync : !stream.resource<external>)

// CHECK: ^bb1(%[[ARG:.+]]: !stream.resource<external>):
^bb1(%arg: !stream.resource<external>):
  // Allocate a new resource in this iteration.
  // CHECK: %[[NEW:.+]], %[[NEW_TP:.+]] = stream.resource.alloca
  %new, %new_tp = stream.resource.alloca uninitialized : !stream.resource<external>{%c4} => !stream.timepoint

  // Execute async operation using block argument as input.
  // No join is created - pass absorbs multiple timepoints directly.
  // CRITICAL: Must use %[[ARG]] (block argument), not %[[INIT]] (initial resource).
  // CHECK: %[[RESULT:.+]], %[[RESULT_TP:.+]] = stream.test.timeline_op await(%[[NEW_TP]], %[[INIT_TP]]) => with(%[[ARG]], %[[NEW]])
  %result, %result_tp = stream.test.timeline_op await(%new_tp) => with(%arg, %new) : (!stream.resource<external>{%c4}, !stream.resource<external>{%c4}) -> !stream.resource<external>{%c4} => !stream.timepoint

  // Await materializes the result.
  // CHECK: %[[SYNC:.+]] = stream.timepoint.await %[[RESULT_TP]] => %[[RESULT]]
  %sync = stream.timepoint.await %result_tp => %result : !stream.resource<external>{%c4}

  // Pass the await result to next iteration or exit.
  // CHECK: cf.cond_br %[[COND]], ^bb1(%[[SYNC]] : !stream.resource<external>), ^bb2(%[[SYNC]] : !stream.resource<external>)
  cf.cond_br %cond, ^bb1(%sync : !stream.resource<external>), ^bb2(%sync : !stream.resource<external>)

^bb2(%exit_result: !stream.resource<external>):
  util.return %exit_result : !stream.resource<external>
}

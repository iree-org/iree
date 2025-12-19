// RUN: iree-opt --split-input-file --iree-stream-elide-timepoints %s | FileCheck %s

// Tests that await is sunk into the else branch when only the else branch uses
// the value directly (yields it without additional operations).

// CHECK-LABEL: @scfIfSinkAwaitIntoElse
util.func public @scfIfSinkAwaitIntoElse(%cond: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0_i32 = arith.constant 0 : i32
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // Produce async value + timepoint.
  // CHECK: %[[CLONE:.+]], %[[TP:.+]] = stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await should be eliminated (sunk into else branch).
  // CHECK-NOT: stream.timepoint.await %[[TP]]
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}

  // scf.if: only else branch uses the awaited value.
  // CHECK: scf.if
  %result = scf.if %cond -> !stream.resource<external> {
    // Then: independent computation, doesn't use clone.
    // CHECK: %[[R0:.+]], %[[TP0:.+]] = stream.test.timeline_op
    %r0, %tp0 = stream.test.timeline_op with() : () -> !stream.resource<external>{%c4} => !stream.timepoint
    %r0_ready = stream.timepoint.await %tp0 => %r0 : !stream.resource<external>{%c4}
    scf.yield %r0_ready : !stream.resource<external>
  } else {
    // Else: yields awaited clone directly - await should be sunk here.
    // CHECK: %[[R0_READY:.+]] = stream.timepoint.await %[[TP]] => %[[CLONE]]
    // CHECK-NEXT: scf.yield %[[R0_READY]]
    scf.yield %awaited : !stream.resource<external>
  }

  util.return %result : !stream.resource<external>
}

// -----

// Tests that await is sunk into the then branch when only the then branch uses
// the value directly.

// CHECK-LABEL: @scfIfSinkAwaitIntoThen
util.func public @scfIfSinkAwaitIntoThen(%cond: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0_i32 = arith.constant 0 : i32
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // Produce async value + timepoint.
  // CHECK: %[[CLONE:.+]], %[[TP:.+]] = stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await should be eliminated (sunk into then branch).
  // CHECK-NOT: stream.timepoint.await %[[TP]]
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}

  // scf.if: only then branch uses the awaited value.
  // CHECK: scf.if
  %result = scf.if %cond -> !stream.resource<external> {
    // Then: yields awaited clone directly - await should be sunk here.
    // CHECK: %[[AWAITED_THEN:.+]] = stream.timepoint.await %[[TP]] => %[[CLONE]]
    // CHECK-NEXT: scf.yield %[[AWAITED_THEN]]
    scf.yield %awaited : !stream.resource<external>
  } else {
    // Else: independent computation.
    // CHECK: %[[R1:.+]], %[[TP1:.+]] = stream.test.timeline_op
    %r1, %tp1 = stream.test.timeline_op with() : () -> !stream.resource<external>{%c4} => !stream.timepoint
    %r1_ready = stream.timepoint.await %tp1 => %r1 : !stream.resource<external>{%c4}
    scf.yield %r1_ready : !stream.resource<external>
  }

  util.return %result : !stream.resource<external>
}

// -----

// Tests that await is sunk into BOTH branches when both branches use the value
// but perform different operations in mutually exclusive execution paths.

// CHECK-LABEL: @scfIfSinkIntoBothBranches
util.func public @scfIfSinkIntoBothBranches(%cond: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // Produce async value + timepoint.
  // CHECK: %{{.+}}, %[[TP:.+]] = stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await should be eliminated (folded into execute in both branches).
  // CHECK-NOT: stream.timepoint.await %[[TP]]
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}

  // Both branches use clone but perform different slices.
  // CHECK: scf.if
  %result = scf.if %cond -> !stream.resource<external> {
    // Then: uses clone for first half slice.
    // CHECK: %[[R0:.+]], %[[TP0:.+]] = stream.test.timeline_op await(%[[TP]]) =>
      %r0, %tp0 = stream.test.timeline_op with(%awaited) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c4} => !stream.timepoint
    %r0_ready = stream.timepoint.await %tp0 => %r0 : !stream.resource<external>{%c4}
    scf.yield %r0_ready : !stream.resource<external>
  } else {
    // Else: uses clone for second half slice.
    // CHECK: %[[R1:.+]], %[[TP1:.+]] = stream.test.timeline_op await(%[[TP]]) =>
      %r1, %tp1 = stream.test.timeline_op with(%awaited) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c4} => !stream.timepoint
    %r1_ready = stream.timepoint.await %tp1 => %r1 : !stream.resource<external>{%c4}
    scf.yield %r1_ready : !stream.resource<external>
  }

  util.return %result : !stream.resource<external>
}

// -----

// Tests that await is NOT sunk when the value is not used in any branch,
// only after the control flow.

// CHECK-LABEL: @scfIfHoistAwaitAfter
util.func public @scfIfHoistAwaitAfter(%cond: i1, %arg0: !stream.resource<external>) -> (!stream.resource<external>, !stream.resource<external>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // Produce async value + timepoint.
  // CHECK: %[[CLONE:.+]], %[[TP:.+]] = stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await stays here since clone isn't used in any branch.
  // CHECK-NOT: stream.timepoint.await %[[TP]]
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}

  // scf.if doesn't use clone at all - produces independent resources.
  // CHECK: %[[BRANCH:.+]] = scf.if %{{.+}} -> (!stream.resource<external>)
  %branch_result = scf.if %cond -> !stream.resource<external> {
    // CHECK: %[[R0:.+]], %[[TP0:.+]] = stream.test.timeline_op
    %r0, %tp0 = stream.test.timeline_op with() : () -> !stream.resource<external>{%c4} => !stream.timepoint
    %r0_ready = stream.timepoint.await %tp0 => %r0 : !stream.resource<external>{%c4}
    scf.yield %r0_ready : !stream.resource<external>
  } else {
    // CHECK: %[[R1:.+]], %[[TP1:.+]] = stream.test.timeline_op
    %r1, %tp1 = stream.test.timeline_op with() : () -> !stream.resource<external>{%c4} => !stream.timepoint
    %r1_ready = stream.timepoint.await %tp1 => %r1 : !stream.resource<external>{%c4}
    scf.yield %r1_ready : !stream.resource<external>
  }

  // clone is used AFTER the control flow - await should stay above the scf.if.
  // CHECK: %[[R1_READY:.+]] = stream.timepoint.await %[[TP]] => %[[CLONE]]
  // CHECK-NEXT: util.return %[[R1_READY]], %[[BRANCH]]
  util.return %awaited, %branch_result : !stream.resource<external>, !stream.resource<external>
}

// -----

// Tests that await + execute is folded into execute's await() clause.

// CHECK-LABEL: @scfIfFoldAwaitWithExecute
util.func public @scfIfFoldAwaitWithExecute(%cond: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index

  // Produce async value + timepoint.
  // CHECK: %{{.+}}, %[[TP:.+]] = stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c16}) -> !stream.resource<external>{%c16} => !stream.timepoint

  // Await should be eliminated (folded into execute below).
  // CHECK-NOT: stream.timepoint.await %[[TP]]
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c16}

  // scf.if produces a resource (not using clone).
  // CHECK: %[[BRANCH:.+]]:2 = scf.if %{{.+}} -> (!stream.resource<external>, !stream.timepoint)
  %branch_resource, %branch_tp = scf.if %cond -> (!stream.resource<external>, !stream.timepoint) {
    // CHECK: %[[R0:.+]], %[[TP0:.+]] = stream.test.timeline_op
    %r0, %tp0 = stream.test.timeline_op with() : () -> !stream.resource<external>{%c16} => !stream.timepoint
    scf.yield %r0, %tp0 : !stream.resource<external>, !stream.timepoint
  } else {
    // CHECK: %[[R1:.+]], %[[TP1:.+]] = stream.test.timeline_op
    %r1, %tp1 = stream.test.timeline_op with() : () -> !stream.resource<external>{%c16} => !stream.timepoint
    scf.yield %r1, %tp1 : !stream.resource<external>, !stream.timepoint
  }

  %branch_ready = stream.timepoint.await %branch_tp => %branch_resource : !stream.resource<external>{%c16}

  // After control flow we use both the clone and branch resource.
  // The awaits are eliminated and timepoints joined, then folded into execute.
  // CHECK: %[[RESULT:.+]], %[[TP_RESULT:.+]] = stream.test.timeline_op await(%[[TP]], %[[BRANCH]]#1) =>
  %result, %result_tp = stream.test.timeline_op with(%branch_ready, %awaited) : (!stream.resource<external>{%c16}, !stream.resource<external>{%c16}) -> !stream.resource<external>{%c16} => !stream.timepoint

  // CHECK: %{{.+}} = stream.timepoint.await %[[TP_RESULT]] => %[[RESULT]]
  %final = stream.timepoint.await %result_tp => %result : !stream.resource<external>{%c16}
  util.return %final : !stream.resource<external>
}

// -----

// Tests that scf.for loops are NOT optimized (conservative).
// Loops may execute 0 times, making sink unsafe.

// CHECK-LABEL: @scfForNoSink
util.func public @scfForNoSink(%arg0: !stream.resource<external>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : index
  %c0_i32 = arith.constant 0 : i32

  // Produce async value + timepoint.
  // CHECK: %{{.+}}, %[[TP:.+]] = stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c4}) -> !stream.resource<external>{%c4} => !stream.timepoint

  // Await is eliminated by folding the timepoint into execute's await clause.
  // The clone timepoint is captured by stream.test.timeline_op in the loop body via await().
  // CHECK-NOT: stream.timepoint.await %[[TP]]
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c4}

  // CHECK: scf.for
  %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%iter = %c0_i32) -> (i32) {
    // Loop body uses clone in stream.test.timeline_op await clause.
    // CHECK: stream.test.timeline_op await(%[[TP]]) =>
    %transfer, %transfer_tp = stream.test.timeline_op with(%awaited) : (!stream.resource<external>{%c4}) -> !stream.resource<staging>{%c4} => !stream.timepoint
    %transfer_ready = stream.timepoint.await %transfer_tp => %transfer : !stream.resource<staging>{%c4}
    %load = stream.async.load %transfer_ready[%c0] : !stream.resource<staging>{%c4} -> i32
    %add = arith.addi %iter, %load : i32
    scf.yield %add : i32
  }

  util.return %result : i32
}

// -----

// Tests that scf.while loops are NOT optimized (conservative).
// While semantics make it unclear which regions execute.

// CHECK-LABEL: @scfWhileNoSink
util.func public @scfWhileNoSink(%arg0: !stream.resource<external>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : index
  %c0_i32 = arith.constant 0 : i32

  // Produce async value + timepoint.
  // CHECK: %{{.+}}, %[[TP:.+]] = stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c4}) -> !stream.resource<external>{%c4} => !stream.timepoint

  // Await is eliminated by folding the timepoint into execute's await clause.
  // CHECK-NOT: stream.timepoint.await %[[TP]]
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c4}

  // CHECK: scf.while
  %result:2 = scf.while (%iter = %c0_i32, %count = %c0) : (i32, index) -> (i32, index) {
    %cond = arith.cmpi slt, %count, %c10 : index
    scf.condition(%cond) %iter, %count : i32, index
  } do {
  ^bb0(%iter: i32, %count: index):
    // While body uses clone in stream.test.timeline_op await clause.
    // CHECK: stream.test.timeline_op await(%[[TP]]) =>
    %transfer, %transfer_tp = stream.test.timeline_op with(%awaited) : (!stream.resource<external>{%c4}) -> !stream.resource<staging>{%c4} => !stream.timepoint
    %transfer_ready = stream.timepoint.await %transfer_tp => %transfer : !stream.resource<staging>{%c4}
    %load = stream.async.load %transfer_ready[%c0] : !stream.resource<staging>{%c4} -> i32
    %add = arith.addi %iter, %load : i32
    %next_count = arith.addi %count, %c1 : index
    scf.yield %add, %next_count : i32, index
  }

  util.return %result#0 : i32
}

// -----

// Tests that scf.index_switch with multiple cases where different cases use the
// async value. Should sink into each case that uses it.

// CHECK-LABEL: @scfIndexSwitchMultipleCases
util.func public @scfIndexSwitchMultipleCases(%selector: index, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // Produce async value + timepoint.
  // CHECK: %{{.+}}, %[[TP:.+]] = stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await should be eliminated (sunk into specific cases).
  // CHECK-NOT: stream.timepoint.await %[[TP]]
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}

  // scf.index_switch: case 0 and 2 use clone, case 1 and default don't.
  // CHECK: scf.index_switch
  %result = scf.index_switch %selector -> !stream.resource<external>
  case 0 {
    // Case 0: uses clone via execute with await.
    // CHECK: %[[R0:.+]], %[[TP0:.+]] = stream.test.timeline_op await(%[[TP]]) =>
      %r0, %tp0 = stream.test.timeline_op with(%awaited) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c4} => !stream.timepoint
    %r0_ready = stream.timepoint.await %tp0 => %r0 : !stream.resource<external>{%c4}
    scf.yield %r0_ready : !stream.resource<external>
  }
  case 1 {
    // Case 1: independent computation, doesn't use clone.
    // CHECK: %[[R1:.+]], %[[TP1:.+]] = stream.test.timeline_op
    %r1, %tp1 = stream.test.timeline_op with() : () -> !stream.resource<external>{%c4} => !stream.timepoint
    %r1_ready = stream.timepoint.await %tp1 => %r1 : !stream.resource<external>{%c4}
    scf.yield %r1_ready : !stream.resource<external>
  }
  case 2 {
    // Case 2: uses clone via execute with await.
    // CHECK: %[[R2:.+]], %[[TP2:.+]] = stream.test.timeline_op await(%[[TP]]) =>
      %r2, %tp2 = stream.test.timeline_op with(%awaited) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c4} => !stream.timepoint
    %r2_ready = stream.timepoint.await %tp2 => %r2 : !stream.resource<external>{%c4}
    scf.yield %r2_ready : !stream.resource<external>
  }
  default {
    // Default: independent computation.
    // CHECK: %[[R_DEFAULT:.+]], %[[TP_DEFAULT:.+]] = stream.test.timeline_op
    %r_default, %tp_default = stream.test.timeline_op with() : () -> !stream.resource<external>{%c4} => !stream.timepoint
    // CHECK: stream.timepoint.await %[[TP_DEFAULT]] => %[[R_DEFAULT]]
    %r_default_ready = stream.timepoint.await %tp_default => %r_default : !stream.resource<external>{%c4}
    scf.yield %r_default_ready : !stream.resource<external>
  }

  util.return %result : !stream.resource<external>
}

// -----

// Tests nested control flow: await optimization applies to outer scf.if,
// and inner operations are handled correctly.

// CHECK-LABEL: @scfIfNested
util.func public @scfIfNested(%outer_cond: i1, %inner_cond: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0_i32 = arith.constant 0 : i32
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  %c8 = arith.constant 8 : index

  // Produce async value + timepoint.
  // CHECK: %[[CLONE:.+]], %[[TP:.+]] = stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await should be eliminated (sunk into else branch).
  // CHECK-NOT: stream.timepoint.await %[[TP]]
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}

  // Outer scf.if: clone used only in else branch.
  // CHECK: %[[OUTER_RESULT:.+]] = scf.if %{{.+}} -> (!stream.resource<external>)
  %outer_result = scf.if %outer_cond -> !stream.resource<external> {
    // Then branch: independent, has nested scf.if.
    // CHECK: %[[INNER_RESULT:.+]] = scf.if %{{.+}} -> (!stream.resource<external>)
    %inner_result = scf.if %inner_cond -> !stream.resource<external> {
      // CHECK: %[[R0:.+]], %[[TP0:.+]] = stream.test.timeline_op
        %r0, %tp0 = stream.test.timeline_op with() : () -> !stream.resource<external>{%c4} => !stream.timepoint
      %r0_ready = stream.timepoint.await %tp0 => %r0 : !stream.resource<external>{%c4}
      scf.yield %r0_ready : !stream.resource<external>
    } else {
      // CHECK: %[[R1:.+]], %[[TP1:.+]] = stream.test.timeline_op
        %r1, %tp1 = stream.test.timeline_op with() : () -> !stream.resource<external>{%c4} => !stream.timepoint
      %r1_ready = stream.timepoint.await %tp1 => %r1 : !stream.resource<external>{%c4}
      scf.yield %r1_ready : !stream.resource<external>
    }
    // CHECK: scf.yield %[[INNER_RESULT]]
    scf.yield %inner_result : !stream.resource<external>
  } else {
    // Else branch: yields clone directly - await should be sunk here.
    // CHECK: %[[R1_READY:.+]] = stream.timepoint.await %[[TP]] => %[[CLONE]]
    // CHECK-NEXT: scf.yield %[[R1_READY]]
    scf.yield %awaited : !stream.resource<external>
  }
  // CHECK: util.return %[[OUTER_RESULT]]
  util.return %outer_result : !stream.resource<external>
}

// -----

// Tests multi-layer nested control flow where await sinking happens at
// multiple nesting levels.

// CHECK-LABEL: @scfIfMultiLayerNested
util.func public @scfIfMultiLayerNested(%outer: i1, %middle: i1, %inner: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // Produce async value + timepoint.
  // CHECK: %[[CLONE:.+]], %[[TP:.+]] = stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await should be eliminated (sunk to innermost level).
  // CHECK-NOT: stream.timepoint.await %[[TP]]
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}

  // Outer scf.if: clone used only in else branch's nested scf.if.
  // CHECK: %[[OUTER_RESULT:.+]] = scf.if %{{.+}} -> (!stream.resource<external>)
  %outer_result = scf.if %outer -> !stream.resource<external> {
    // Outer then: independent.
    // CHECK: %[[R0:.+]], %[[TP0:.+]] = stream.test.timeline_op
    %r0, %tp0 = stream.test.timeline_op with() : () -> !stream.resource<external>{%c4} => !stream.timepoint
    %r0_ready = stream.timepoint.await %tp0 => %r0 : !stream.resource<external>{%c4}
    scf.yield %r0_ready : !stream.resource<external>
  } else {
    // Outer else: has middle scf.if.
    // CHECK: %[[MIDDLE_RESULT:.+]] = scf.if %{{.+}} -> (!stream.resource<external>)
    %middle_result = scf.if %middle -> !stream.resource<external> {
      // Middle then: independent.
      // CHECK: %[[R_MID:.+]], %[[TP_MID:.+]] = stream.test.timeline_op
        %r_mid, %tp_mid = stream.test.timeline_op with() : () -> !stream.resource<external>{%c4} => !stream.timepoint
      // CHECK: stream.timepoint.await %[[TP_MID]] => %[[R_MID]]
      %r_mid_ready = stream.timepoint.await %tp_mid => %r_mid : !stream.resource<external>{%c4}
      scf.yield %r_mid_ready : !stream.resource<external>
    } else {
      // Middle else: has inner scf.if where clone is used.
      // CHECK: %[[INNER_RESULT:.+]] = scf.if %{{.+}} -> (!stream.resource<external>)
      %inner_result = scf.if %inner -> !stream.resource<external> {
        // Inner then: uses clone - await sunk to innermost level.
        // CHECK: %[[R:.+]], %[[TP2:.+]] = stream.test.timeline_op await(%[[TP]]) =>
                  %r2, %tp2 = stream.test.timeline_op with(%awaited) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c4} => !stream.timepoint
        // CHECK: %[[R2_READY:.+]] = stream.timepoint.await %[[TP2]] => %[[R]]
        %r2_ready = stream.timepoint.await %tp2 => %r2 : !stream.resource<external>{%c4}
        // CHECK: scf.yield %[[R2_READY]]
        scf.yield %r2_ready : !stream.resource<external>
      } else {
        // Inner else: yields clone directly - await sunk here too.
        // CHECK: %[[R2_READY:.+]] = stream.timepoint.await %[[TP]] => %[[CLONE]]
        // CHECK-NEXT: scf.yield %[[R2_READY]]
        scf.yield %awaited : !stream.resource<external>
      }
      // CHECK: scf.yield %[[INNER_RESULT]]
      scf.yield %inner_result : !stream.resource<external>
    }
    // CHECK: scf.yield %[[MIDDLE_RESULT]]
    scf.yield %middle_result : !stream.resource<external>
  }
  // CHECK: util.return %[[OUTER_RESULT]]
  util.return %outer_result : !stream.resource<external>
}

// -----

// Tests multiple timepoints at different nesting levels.

// CHECK-LABEL: @scfIfMultipleTimepointsNested
util.func public @scfIfMultipleTimepointsNested(%outer: i1, %inner: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // First async operation.
  // CHECK: %[[CLONE1:.+]], %[[TP1:.+]] = stream.test.timeline_op
  %clone1, %tp1 = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await should be eliminated.
  // CHECK-NOT: stream.timepoint.await %[[TP1]]
  %awaited1 = stream.timepoint.await %tp1 => %clone1 : !stream.resource<external>{%c8}

  // CHECK: %[[OUTER_RESULT:.+]] = scf.if %{{.+}} -> (!stream.resource<external>)
  %outer_result = scf.if %outer -> !stream.resource<external> {
    // Inner async operation in then branch.
    // CHECK: %[[CLONE2:.+]], %[[TP2:.+]] = stream.test.timeline_op await(%[[TP1]]) =>
    %clone2, %tp2 = stream.test.timeline_op with(%awaited1) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

    // CHECK-NOT: stream.timepoint.await %[[TP2]]
    %awaited2 = stream.timepoint.await %tp2 => %clone2 : !stream.resource<external>{%c8}

    // CHECK: %[[INNER_RESULT:.+]] = scf.if %{{.+}} -> (!stream.resource<external>)
    %inner_result = scf.if %inner -> !stream.resource<external> {
      // Uses clone2 from parent region.
      // CHECK: %[[R:.+]], %[[TP_R:.+]] = stream.test.timeline_op await(%[[TP2]]) =>
      %r, %tp = stream.test.timeline_op with(%awaited2) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c4} => !stream.timepoint
      // CHECK: %[[R_READY:.+]] = stream.timepoint.await %[[TP_R]] => %[[R]]
      %r_ready = stream.timepoint.await %tp => %r : !stream.resource<external>{%c4}
      // CHECK: scf.yield %[[R_READY]]
      scf.yield %r_ready : !stream.resource<external>
    } else {
      // CHECK: %[[R_ELSE:.+]], %[[TP_R_ELSE:.+]] = stream.test.timeline_op
        %r, %tp = stream.test.timeline_op with() : () -> !stream.resource<external>{%c4} => !stream.timepoint
      // CHECK: %[[R_ELSE_READY:.+]] = stream.timepoint.await %[[TP_R_ELSE]] => %[[R_ELSE]]
      %r_ready = stream.timepoint.await %tp => %r : !stream.resource<external>{%c4}
      // CHECK: scf.yield %[[R_ELSE_READY]]
      scf.yield %r_ready : !stream.resource<external>
    }
    // CHECK: scf.yield %[[INNER_RESULT]]
    scf.yield %inner_result : !stream.resource<external>
  } else {
    // Uses clone1 from outer region.
    // CHECK: %[[R3:.+]], %[[TP3:.+]] = stream.test.timeline_op await(%[[TP1]]) =>
    %r3, %tp3 = stream.test.timeline_op with(%awaited1) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c4} => !stream.timepoint
    %r3_ready = stream.timepoint.await %tp3 => %r3 : !stream.resource<external>{%c4}
    scf.yield %r3_ready : !stream.resource<external>
  }
  // CHECK: util.return %[[OUTER_RESULT]]
  util.return %outer_result : !stream.resource<external>
}

// -----

// Tests that await on a joined timepoint can be folded into execute operations
// in scf.if branches. The join should be absorbed into the execute's await clause.

// CHECK-LABEL: @scfIfAwaitJoinedTimepoint
util.func public @scfIfAwaitJoinedTimepoint(%cond: i1, %arg0: !stream.resource<external>, %arg1: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // Two independent async operations.
  // CHECK: %[[CLONE1:.+]], %[[TP1:.+]] = stream.test.timeline_op
  %clone1, %tp1 = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // CHECK: %[[CLONE2:.+]], %[[TP2:.+]] = stream.test.timeline_op
  %clone2, %tp2 = stream.test.timeline_op with(%arg1) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Join the two timepoints.
  // CHECK: %[[JOINED:.+]] = stream.timepoint.join max(%[[TP1]], %[[TP2]])
  %joined = stream.timepoint.join max(%tp1, %tp2) => !stream.timepoint

  // Await the joined timepoint - should be eliminated (folded into execute ops).
  // CHECK-NOT: stream.timepoint.await
  %awaited1 = stream.timepoint.await %joined => %clone1 : !stream.resource<external>{%c8}
  %awaited2 = stream.timepoint.await %joined => %clone2 : !stream.resource<external>{%c8}

  // scf.if: then uses awaited1, else uses awaited2.
  // CHECK: scf.if
  %result = scf.if %cond -> !stream.resource<external> {
    // Then: uses awaited1 (clone1 synchronized via joined timepoint).
    // The joined timepoint should be absorbed into this execute's await clause.
    // CHECK: %[[R_THEN:.+]], %[[TP_THEN:.+]] = stream.test.timeline_op await(%[[JOINED]]) =>
    %r, %tp = stream.test.timeline_op with(%awaited1) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c4} => !stream.timepoint
    // CHECK: %[[R_READY_THEN:.+]] = stream.timepoint.await %[[TP_THEN]] => %[[R_THEN]]
    %r_ready = stream.timepoint.await %tp => %r : !stream.resource<external>{%c4}
    // CHECK: scf.yield %[[R_READY_THEN]]
    scf.yield %r_ready : !stream.resource<external>
  } else {
    // Else: uses awaited2 (clone2 synchronized via joined timepoint).
    // CHECK: %[[R_ELSE:.+]], %[[TP_ELSE:.+]] = stream.test.timeline_op await(%[[JOINED]]) =>
    %r, %tp = stream.test.timeline_op with(%awaited2) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c4} => !stream.timepoint
    // CHECK: %[[R_READY_ELSE:.+]] = stream.timepoint.await %[[TP_ELSE]] => %[[R_ELSE]]
    %r_ready = stream.timepoint.await %tp => %r : !stream.resource<external>{%c4}
    // CHECK: scf.yield %[[R_READY_ELSE]]
    scf.yield %r_ready : !stream.resource<external>
  }

  util.return %result : !stream.resource<external>
}

// -----

// Tests that multiple independent awaits are sunk selectively into the branches
// that actually use them. Await1 should sink into then, await2 into else.

// CHECK-LABEL: @scfIfMultipleIndependentAwaits
util.func public @scfIfMultipleIndependentAwaits(%cond: i1, %arg0: !stream.resource<external>, %arg1: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // First async operation.
  // CHECK: %[[CLONE1:.+]], %[[TP1:.+]] = stream.test.timeline_op
  %clone1, %tp1 = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Second async operation.
  // CHECK: %[[CLONE2:.+]], %[[TP2:.+]] = stream.test.timeline_op
  %clone2, %tp2 = stream.test.timeline_op with(%arg1) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await both - these should be eliminated (sunk into respective branches).
  // CHECK-NOT: stream.timepoint.await %[[TP1]]
  // CHECK-NOT: stream.timepoint.await %[[TP2]]
  %awaited1 = stream.timepoint.await %tp1 => %clone1 : !stream.resource<external>{%c8}
  %awaited2 = stream.timepoint.await %tp2 => %clone2 : !stream.resource<external>{%c8}

  // scf.if: then uses only awaited1, else uses only awaited2.
  // CHECK: scf.if
  %result = scf.if %cond -> !stream.resource<external> {
    // Then: uses awaited1 only - await1 should be folded here.
    // CHECK: %[[R_THEN:.+]], %[[TP_THEN:.+]] = stream.test.timeline_op await(%[[TP1]]) =>
    %r, %tp = stream.test.timeline_op with(%awaited1) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c4} => !stream.timepoint
    // CHECK: stream.timepoint.await %[[TP_THEN]] => %[[R_THEN]]
    %r_ready = stream.timepoint.await %tp => %r : !stream.resource<external>{%c4}
    scf.yield %r_ready : !stream.resource<external>
  } else {
    // Else: uses awaited2 only - await2 should be folded here.
    // CHECK: %[[R_ELSE:.+]], %[[TP_ELSE:.+]] = stream.test.timeline_op await(%[[TP2]]) =>
    %r, %tp = stream.test.timeline_op with(%awaited2) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c4} => !stream.timepoint
    // CHECK: stream.timepoint.await %[[TP_ELSE]] => %[[R_ELSE]]
    %r_ready = stream.timepoint.await %tp => %r : !stream.resource<external>{%c4}
    scf.yield %r_ready : !stream.resource<external>
  }

  util.return %result : !stream.resource<external>
}

// -----

// Tests that awaited values used as loop-carried iter_args are handled conservatively.
// The await should NOT be sunk into the loop because the value is used as an iter_arg.

// CHECK-LABEL: @scfForLoopCarriedAwaitedValue
util.func public @scfForLoopCarriedAwaitedValue(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c10 = arith.constant 10 : index

  // Async operation producing initial value.
  // CHECK: %[[CLONE:.+]], %[[TP:.+]] = stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await before loop - conservative: should remain at function level.
  // Cannot sink into loop because value is used as iter_arg.
  // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[TP]] => %[[CLONE]]
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}

  // Loop uses awaited value as initial iter_arg and transforms it each iteration.
  // CHECK: %[[LOOP_RESULT:.+]] = scf.for %{{.+}} = %c0 to %c10 step %c1
  // CHECK-SAME: iter_args(%{{.+}} = %[[AWAITED]])
  %loop_result = scf.for %i = %c0 to %c10 step %c1
    iter_args(%iter = %awaited) -> (!stream.resource<external>) {
    // Each iteration produces a new value based on previous iteration.
    %next, %next_tp = stream.test.timeline_op with(%iter) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint
    %next_awaited = stream.timepoint.await %next_tp => %next : !stream.resource<external>{%c8}
    scf.yield %next_awaited : !stream.resource<external>
  }
  // CHECK: util.return %[[LOOP_RESULT]]
  util.return %loop_result : !stream.resource<external>
}

// -----

// Tests that an await operation defined outside an scf.for loop can be folded
// into a stream.test.timeline_op operation within a nested scf.if branch inside the loop.
// This demonstrates the pass's ability to optimize across loop boundaries when the
// await can be absorbed into an execute's await clause.

// CHECK-LABEL: @scfIfInsideScfFor
util.func public @scfIfInsideScfFor(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c10 = arith.constant 10 : index
  %true = arith.constant true

  // Async operation producing value used inside loop.
  // CHECK: %{{.+}}, %[[TP:.+]] = stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await before loop - should be eliminated (folded into execute inside loop).
  // CHECK-NOT: stream.timepoint.await %[[TP]]
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}

  // Loop with conditional use of awaited value.
  // CHECK: %[[LOOP_RESULT:.+]] = scf.for {{.+}} iter_args(%[[ITER:.+]] = {{.+}})
  %loop_result = scf.for %i = %c0 to %c10 step %c1
    iter_args(%iter = %arg0) -> (!stream.resource<external>) {

    // Conditional inside loop uses awaited value from outside loop.
    // After propagate-timepoints, scf.if yields both resource and timepoint.
    // CHECK: %{{.+}}:2 = scf.if %{{.+}} -> (!stream.resource<external>, !stream.timepoint)
    %inner, %inner_tp = scf.if %true -> (!stream.resource<external>, !stream.timepoint) {
      // Then: await is folded into this execute's await clause.
      // CHECK: %[[R:.+]], %[[TP0:.+]] = stream.test.timeline_op await(%[[TP]]) =>
            %r0, %tp0 = stream.test.timeline_op with(%awaited) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c4} => !stream.timepoint
      // The await is redundant since scf.if yields the timepoint.
      // ElideTimepointsPass now eliminates this redundancy by yielding the
      // original resource directly instead of the awaited value.
      // CHECK-NOT: scf.yield %{{.*stream.timepoint.await.*}}, %[[TP0]]
      %r0_ready = stream.timepoint.await %tp0 => %r0 : !stream.resource<external>{%c4}
      // CHECK: scf.yield %[[R]], %[[TP0]]
      scf.yield %r0_ready, %tp0 : !stream.resource<external>, !stream.timepoint
    } else {
      // Else: yields iter_arg and immediate timepoint.
      // CHECK: %[[IMM:.+]] = stream.timepoint.immediate
      %imm = stream.timepoint.immediate => !stream.timepoint
      // CHECK: scf.yield %[[ITER]], %[[IMM]]
      scf.yield %iter, %imm : !stream.resource<external>, !stream.timepoint
    }
    // Await the yielded timepoint before next iteration.
    %inner_ready = stream.timepoint.await %inner_tp => %inner : !stream.resource<external>{%c4}
    scf.yield %inner_ready : !stream.resource<external>
  }
  // CHECK: util.return %[[LOOP_RESULT]]
  util.return %loop_result : !stream.resource<external>
}

// -----

// Tests that timepoints yielded from scf.if branches can be awaited after the
// control flow, and coverage is correctly propagated.

// CHECK-LABEL: @scfIfYieldingTimepoint
util.func public @scfIfYieldingTimepoint(%cond: i1, %arg0: !stream.resource<external>, %arg1: !stream.resource<external>) -> !stream.resource<external> {
  %c8 = arith.constant 8 : index

  // scf.if yields different resources and timepoints from each branch.
  // Both branches yield same-sized resources to avoid type complexity.
  // CHECK: %[[RESULT:.+]]:2 = scf.if %{{.+}} -> (!stream.resource<external>, !stream.timepoint)
  %result, %result_tp = scf.if %cond -> (!stream.resource<external>, !stream.timepoint) {
    // Then: produces clone of arg0 + timepoint.
    // CHECK: %[[CLONE_THEN:.+]], %[[CLONE_TP:.+]] = stream.test.timeline_op
    %clone, %clone_tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint
    // CHECK: scf.yield %[[CLONE_THEN]], %[[CLONE_TP]]
    scf.yield %clone, %clone_tp : !stream.resource<external>, !stream.timepoint
  } else {
    // Else: produces clone of arg1 + timepoint.
    // CHECK: %{{.+}}, %{{.+}} = stream.test.timeline_op
    %clone, %clone_tp = stream.test.timeline_op with(%arg1) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint
    scf.yield %clone, %clone_tp : !stream.resource<external>, !stream.timepoint
  }

  // Await the yielded timepoint - should be sunk or folded.
  // CHECK-NOT: stream.timepoint.await %[[RESULT]]#1
  %awaited = stream.timepoint.await %result_tp => %result : !stream.resource<external>{%c8}

  // Use the awaited value in another async operation.
  // The yielded timepoint should be absorbed into this execute's await clause.
  // CHECK: %[[FINAL:.+]], %[[TP_FINAL:.+]] = stream.test.timeline_op await(%[[RESULT]]#1) =>
  %final, %final_tp = stream.test.timeline_op with(%awaited) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint
  // NOTE: This await is required because the function is public and returns only a resource.
  // Public function signatures cannot be changed to return (resource, timepoint) tuples.
  // CHECK: stream.timepoint.await %[[TP_FINAL]] => %[[FINAL]]
  %final_awaited = stream.timepoint.await %final_tp => %final : !stream.resource<external>{%c8}

  util.return %final_awaited : !stream.resource<external>
}

// -----

// Tests that scf.for nested inside scf.if is handled correctly: await should
// sink into the if branch, then be conservative inside the loop.

// CHECK-LABEL: @scfForInsideScfIf
util.func public @scfForInsideScfIf(%cond: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c10 = arith.constant 10 : index

  // Async operation producing value.
  // CHECK: %[[CLONE:.+]], %[[TP:.+]] = stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await should be eliminated (sunk into then branch, before loop).
  // CHECK-NOT: stream.timepoint.await %[[TP]]
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}

  // scf.if: only then branch uses the awaited value (in a nested loop).
  // CHECK: scf.if
  %result = scf.if %cond -> !stream.resource<external> {
    // Then: has loop that uses awaited value.
    // Await should be sunk here (into the then branch, before the loop).
    // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[TP]] => %[[CLONE]]
    // CHECK: %[[LOOP_RESULT:.+]] = scf.for %{{.+}} = %c0 to %c10 step %c1
    // CHECK-SAME: iter_args(%{{.+}} = %[[AWAITED]])
    %loop_result = scf.for %i = %c0 to %c10 step %c1
      iter_args(%iter = %awaited) -> (!stream.resource<external>) {
      %next, %next_tp = stream.test.timeline_op with(%iter) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint
      // TODO: This await is redundant. The scf.for should yield (resource, timepoint) and
      // use timepoint-carrying iter_args, eliminating the need for this await.
      // Part 2 will fix propagate-timepoints to handle loop-carried timepoints.
      %next_awaited = stream.timepoint.await %next_tp => %next : !stream.resource<external>{%c8}
      scf.yield %next_awaited : !stream.resource<external>
    }
    // CHECK: scf.yield %[[LOOP_RESULT]]
    scf.yield %loop_result : !stream.resource<external>
  } else {
    // Else: independent computation.
    // CHECK: %[[R1:.+]], %[[TP1:.+]] = stream.test.timeline_op
    %r1, %tp1 = stream.test.timeline_op with() : () -> !stream.resource<external>{%c4} => !stream.timepoint
    // CHECK: %[[R1_READY:.+]] = stream.timepoint.await %[[TP1]] => %[[R1]]
    %r1_ready = stream.timepoint.await %tp1 => %r1 : !stream.resource<external>{%c4}
    // CHECK: scf.yield %[[R1_READY]]
    scf.yield %r1_ready : !stream.resource<external>
  }

  util.return %result : !stream.resource<external>
}

// -----

// Tests complex coverage chains in scf.if: multiple timepoints with joins
// and transitive dependencies. The pass must analyze coverage relationships
// across branches to determine minimal await sets.

// CHECK-LABEL: @scfIfComplexCoverageChain
util.func public @scfIfComplexCoverageChain(%cond: i1, %arg0: !stream.resource<external>, %arg1: !stream.resource<external>, %arg2: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // Three independent async operations.
  // CHECK: %[[CLONE1:.+]], %[[TP1:.+]] = stream.test.timeline_op
  %clone1, %tp1 = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // CHECK: %[[CLONE2:.+]], %[[TP2:.+]] = stream.test.timeline_op
  %clone2, %tp2 = stream.test.timeline_op with(%arg1) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // CHECK: %[[CLONE3:.+]], %[[TP3:.+]] = stream.test.timeline_op
  %clone3, %tp3 = stream.test.timeline_op with(%arg2) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Join tp1 and tp2 to create coverage chain.
  %join12 = stream.timepoint.join max(%tp1, %tp2) => !stream.timepoint

  // Await individual timepoints.
  // CHECK-NOT: stream.timepoint.await %[[TP1]]
  // CHECK-NOT: stream.timepoint.await %[[TP3]]
  %awaited1 = stream.timepoint.await %tp1 => %clone1 : !stream.resource<external>{%c8}
  %awaited3 = stream.timepoint.await %tp3 => %clone3 : !stream.resource<external>{%c8}

  // scf.if: then uses awaited1 and awaited3, else uses only awaited1.
  // The pass must determine optimal await sets for each branch.
  // CHECK: scf.if
  %result = scf.if %cond -> !stream.resource<external> {
    // Then: uses both awaited1 and awaited3.
    // Pass creates fresh join of tp1 and tp3 inside branch.
    // CHECK: %[[R:.+]], %[[TP_R:.+]] = stream.test.timeline_op await(%[[TP1]], %[[TP3]]) =>
    %r, %tp = stream.test.timeline_op with(%awaited1, %awaited3) : (!stream.resource<external>{%c8}, !stream.resource<external>{%c8}) -> !stream.resource<external>{%c4} => !stream.timepoint
    // CHECK: stream.timepoint.await %[[TP_R]] => %[[R]]
    %r_ready = stream.timepoint.await %tp => %r : !stream.resource<external>{%c4}
    scf.yield %r_ready : !stream.resource<external>
  } else {
    // Else: uses only awaited1.
    // Only tp1 needs to be awaited.
    // CHECK: %[[R:.+]], %[[TP_R:.+]] = stream.test.timeline_op await(%[[TP1]]) =>
    %r, %tp = stream.test.timeline_op with(%awaited1) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c4} => !stream.timepoint
    %r_ready = stream.timepoint.await %tp => %r : !stream.resource<external>{%c4}
    scf.yield %r_ready : !stream.resource<external>
  }

  util.return %result : !stream.resource<external>
}

// -----

// Tests partial coverage: tp2 transitively covers tp1 in the then branch
// (because tp2's work depends on tp1), but in the else branch tp2 does not
// cover tp1. This tests the pass's coverage analysis across branches.

// CHECK-LABEL: @scfIfPartialCoverage
util.func public @scfIfPartialCoverage(%cond: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // First async operation.
  // CHECK: %[[CLONE1:.+]], %[[TP1:.+]] = stream.test.timeline_op
  %clone1, %tp1 = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Second async operation that depends on first (tp2 covers tp1).
  // CHECK: %[[CLONE2:.+]], %[[TP2:.+]] = stream.test.timeline_op await(%[[TP1]]) =>
  %clone2, %tp2 = stream.test.timeline_op await(%tp1) => with(%clone1) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await both timepoints.
  // CHECK-NOT: stream.timepoint.await %[[TP1]]
  // CHECK-NOT: stream.timepoint.await %[[TP2]]
  %awaited1 = stream.timepoint.await %tp1 => %clone1 : !stream.resource<external>{%c8}
  %awaited2 = stream.timepoint.await %tp2 => %clone2 : !stream.resource<external>{%c8}

  // scf.if: then uses both awaited1 and awaited2, else uses only awaited1.
  // Since tp2 depends on tp1 (tp2's execute awaits tp1), the pass recognizes
  // that tp1 is sufficient coverage for both resources in the then branch.
  // CHECK: scf.if
  %result = scf.if %cond -> !stream.resource<external> {
    // Then: uses both awaited1 and awaited2.
    // Pass recognizes tp1 is sufficient since tp2 depends on tp1.
    // CHECK: %[[R:.+]], %[[TP_R:.+]] = stream.test.timeline_op await(%[[TP1]]) =>
    %r, %tp = stream.test.timeline_op with(%awaited1, %awaited2) : (!stream.resource<external>{%c8}, !stream.resource<external>{%c8}) -> !stream.resource<external>{%c4} => !stream.timepoint
    // CHECK: stream.timepoint.await %[[TP_R]] => %[[R]]
    %r_ready = stream.timepoint.await %tp => %r : !stream.resource<external>{%c4}
    scf.yield %r_ready : !stream.resource<external>
  } else {
    // Else: uses only awaited1.
    // Only tp1 needs to be awaited.
    // CHECK: %[[R:.+]], %[[TP_R:.+]] = stream.test.timeline_op await(%[[TP1]]) =>
    %r, %tp = stream.test.timeline_op with(%awaited1) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c4} => !stream.timepoint
    // CHECK: stream.timepoint.await %[[TP_R]] => %[[R]]
    %r_ready = stream.timepoint.await %tp => %r : !stream.resource<external>{%c4}
    scf.yield %r_ready : !stream.resource<external>
  }

  util.return %result : !stream.resource<external>
}

// -----

// Tests that awaits before scf.while are handled conservatively.
// The await should remain before the while loop because the value is used
// in the condition region, which executes before each iteration.

// CHECK-LABEL: @scfWhileAwaitInCondition
util.func public @scfWhileAwaitInCondition(%arg0: !stream.resource<external>, %limit: index) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index

  // Async operation producing value.
  // CHECK: %[[CLONE:.+]], %[[TP:.+]] = stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // Await before while loop - should remain at function level.
  // Cannot sink into loop because value may be used in condition region.
  // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[TP]] => %[[CLONE]]
  %awaited = stream.timepoint.await %tp => %clone : !stream.resource<external>{%c8}

  // While loop that uses awaited value.
  // CHECK: %[[WHILE_RESULT:.+]]:2 = scf.while (%[[I:.+]] = %c0, %[[ITER:.+]] = %[[AWAITED]])
  %result:2 = scf.while (%i = %c0, %iter = %awaited) : (index, !stream.resource<external>) -> (index, !stream.resource<external>) {
    // Condition region: check if we should continue.
    // CHECK: arith.cmpi slt, %[[I]]
    %cond = arith.cmpi slt, %i, %limit : index
    // CHECK: scf.condition(%{{.+}}) %[[I]], %[[ITER]]
    scf.condition(%cond) %i, %iter : index, !stream.resource<external>
  } do {
  ^bb0(%i_body: index, %iter_body: !stream.resource<external>):
    // Body region: transform the resource.
    %next, %next_tp = stream.test.timeline_op with(%iter_body) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint
    %next_awaited = stream.timepoint.await %next_tp => %next : !stream.resource<external>{%c8}
    %next_i = arith.addi %i_body, %c1 : index
    scf.yield %next_i, %next_awaited : index, !stream.resource<external>
  }
  // CHECK: util.return %[[WHILE_RESULT]]#1
  util.return %result#1 : !stream.resource<external>
}

// -----

// Tests that timeline-aware operations (those implementing TimelineAwareOpInterface)
// inside scf.if are handled correctly. These operations participate in timeline
// scheduling and don't need explicit awaits before them.

// CHECK-LABEL: @scfIfTimelineAwareOp
util.func public @scfIfTimelineAwareOp(%cond: i1, %arg0: !stream.resource<external>, %signal_fence: !stream.test.fence) -> !stream.resource<external> {
  %c8 = arith.constant 8 : index

  // Async operation producing value.
  // CHECK: %[[CLONE:.+]], %[[TP:.+]] = stream.test.timeline_op
  %clone, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // scf.if with timeline-aware operation in then branch.
  // CHECK: scf.if
  %result = scf.if %cond -> !stream.resource<external> {
    // Then: uses timeline-aware test op.
    // The timepoint should be exported to fence and passed to timeline-aware op.
    // CHECK-NOT: stream.timepoint.await %[[TP]]
    // CHECK: %[[WAIT_FENCE:.+]] = stream.timepoint.export %[[TP]]
    %wait_fence = stream.timepoint.export %tp => (!stream.test.fence)

    // Timeline-aware op - should NOT have await before it.
    // CHECK: %[[AWARE:.+]] = stream.test.timeline_aware(%[[CLONE]]) waits(%[[WAIT_FENCE]]) signals(%{{.+}})
    %aware = stream.test.timeline_aware(%clone) waits(%wait_fence) signals(%signal_fence) : (!stream.resource<external>) -> !stream.resource<external>

    // Import signal fence back to timepoint.
    // CHECK: %[[AWARE_TP:.+]] = stream.timepoint.import
    %aware_tp = stream.timepoint.import %signal_fence : (!stream.test.fence) => !stream.timepoint

    // Subsequent async.execute should await imported timepoint.
    // CHECK: %{{.+}}, %{{.+}} = stream.test.timeline_op await(%[[AWARE_TP]]) => with(%[[AWARE]])
    %awaited = stream.timepoint.await %aware_tp => %aware : !stream.resource<external>{%c8}
    %final, %final_tp = stream.test.timeline_op with(%awaited) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

    scf.yield %final : !stream.resource<external>
  } else {
    // Else: just return arg0.
    scf.yield %arg0 : !stream.resource<external>
  }

  util.return %result : !stream.resource<external>
}

// -----

// Tests that multiple resources from the same await can be yielded without
// their awaits when the timepoint is also yielded.

// CHECK-LABEL: @scfMultipleAwaitResultsYielded
util.func public @scfMultipleAwaitResultsYielded(%cond: i1, %arg0: !stream.resource<external>) -> (!stream.resource<external>, !stream.resource<external>) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // Async execute producing two results.
  // CHECK: %[[RESULTS:.+]]:2, %[[TP:.+]] = stream.test.timeline_op
  %r1, %r2, %tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> (!stream.resource<external>{%c4}, !stream.resource<external>{%c4}) => !stream.timepoint

  // scf.if yields both awaited resources and the timepoint.
  // Both awaits should be eliminated since the timepoint is yielded.
  // CHECK: %{{.+}}:3 = scf.if %{{.+}} -> (!stream.resource<external>, !stream.resource<external>, !stream.timepoint)
  %if_r1, %if_r2, %if_tp = scf.if %cond -> (!stream.resource<external>, !stream.resource<external>, !stream.timepoint) {
    %awaited1 = stream.timepoint.await %tp => %r1 : !stream.resource<external>{%c4}
    %awaited2 = stream.timepoint.await %tp => %r2 : !stream.resource<external>{%c4}
    // CHECK: scf.yield %[[RESULTS]]#0, %[[RESULTS]]#1, %[[TP]]
    scf.yield %awaited1, %awaited2, %tp : !stream.resource<external>, !stream.resource<external>, !stream.timepoint
  } else {
    // CHECK: %[[IMM:.+]] = stream.timepoint.immediate
    %imm = stream.timepoint.immediate => !stream.timepoint
    // CHECK: scf.yield %[[RESULTS]]#0, %[[RESULTS]]#1, %[[IMM]]
    scf.yield %r1, %r2, %imm : !stream.resource<external>, !stream.resource<external>, !stream.timepoint
  }
  // The awaits use joined timepoints that cover both %if_tp and the original %tp.
  %final1 = stream.timepoint.await %if_tp => %if_r1 : !stream.resource<external>{%c4}
  %final2 = stream.timepoint.await %if_tp => %if_r2 : !stream.resource<external>{%c4}
  util.return %final1, %final2 : !stream.resource<external>, !stream.resource<external>
}

// -----

// Tests scf.while loop with await elimination when timepoint is yielded.

// CHECK-LABEL: @scfWhileAwaitElimination
util.func public @scfWhileAwaitElimination(%arg0: !stream.resource<external>, %count: index) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index

  // Initial async operation.
  // CHECK: %[[INIT:.+]], %[[INIT_TP:.+]] = stream.test.timeline_op
  %init, %init_tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // scf.while loop that yields resource and timepoint.
  // Awaits inside the loop body should be eliminated.
  // CHECK: %[[WHILE:.+]]:3 = scf.while
  // CHECK-SAME: (%[[I:.+]] = %c0, %[[R:.+]] = %[[INIT]], %[[TP:.+]] = %[[INIT_TP]])
  %while_i, %while_r, %while_tp = scf.while (%i = %c0, %r = %init, %tp = %init_tp)
    : (index, !stream.resource<external>, !stream.timepoint)
    -> (index, !stream.resource<external>, !stream.timepoint) {
    // CHECK: arith.cmpi slt, %[[I]]
    %cond = arith.cmpi slt, %i, %count : index
    // CHECK: scf.condition(%{{.+}}) %[[I]], %[[R]], %[[TP]]
    scf.condition(%cond) %i, %r, %tp : index, !stream.resource<external>, !stream.timepoint
  } do {
  ^bb0(%i: index, %r: !stream.resource<external>, %tp: !stream.timepoint):
    // Await is redundant because we yield the resource and timepoint together.
    %awaited = stream.timepoint.await %tp => %r : !stream.resource<external>{%c8}

    // Clone the awaited resource.
    // CHECK: %[[CLONED:.+]], %[[CLONED_TP:.+]] = stream.test.timeline_op await(%[[TP]]) => with(%[[R]])
    %cloned, %cloned_tp = stream.test.timeline_op with(%awaited) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

    // Await here is also redundant.
    // CHECK: %[[CLONED_AWAITED:.+]] = stream.timepoint.await %[[CLONED_TP]] => %[[CLONED]]
    %cloned_awaited = stream.timepoint.await %cloned_tp => %cloned : !stream.resource<external>{%c8}
    // CHECK: %[[NEXT_I:.+]] = arith.addi %[[I]]
    %next_i = arith.addi %i, %c1 : index
    // CHECK: scf.yield %[[NEXT_I]], %[[CLONED_AWAITED]], %[[CLONED_TP]]
    scf.yield %next_i, %cloned_awaited, %cloned_tp : index, !stream.resource<external>, !stream.timepoint
  }
  // CHECK: %[[FINAL:.+]] = stream.timepoint.await %[[WHILE]]#2 => %[[WHILE]]#1
  %final = stream.timepoint.await %while_tp => %while_r : !stream.resource<external>{%c8}
  // CHECK: util.return %[[FINAL]]
  util.return %final : !stream.resource<external>
}

// -----

// Tests mixed pattern: some resources awaited, some not, all yielded with
// same timepoint. Only the awaited ones should have their awaits eliminated.

// CHECK-LABEL: @scfMixedAwaitedAndNonAwaited
util.func public @scfMixedAwaitedAndNonAwaited(%cond: i1, %arg0: !stream.resource<external>, %arg1: !stream.resource<external>)
  -> (!stream.resource<external>, !stream.resource<external>) {
  %c8 = arith.constant 8 : index

  // Async operation on arg0.
  // CHECK: %[[R1:.+]], %[[TP1:.+]] = stream.test.timeline_op
  %r1, %tp1 = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c8}) -> !stream.resource<external>{%c8} => !stream.timepoint

  // scf.if yields: awaited r1, non-awaited arg1, and tp1.
  // Only r1's await should be eliminated.
  // CHECK: %[[IF:.+]]:3 = scf.if %{{.+}} -> (!stream.resource<external>, !stream.resource<external>, !stream.timepoint)
  %if_r1, %if_r2, %if_tp = scf.if %cond -> (!stream.resource<external>, !stream.resource<external>, !stream.timepoint) {
    %awaited_r1 = stream.timepoint.await %tp1 => %r1 : !stream.resource<external>{%c8}
    // arg1 is not awaited, just passed through.
    scf.yield %awaited_r1, %arg1, %tp1 : !stream.resource<external>, !stream.resource<external>, !stream.timepoint
  } else {
    %imm = stream.timepoint.immediate => !stream.timepoint
    scf.yield %arg0, %arg1, %imm : !stream.resource<external>, !stream.resource<external>, !stream.timepoint
  }
  // The awaits may use joined timepoints and reference different resources.
  %final1 = stream.timepoint.await %if_tp => %if_r1 : !stream.resource<external>{%c8}
  // CHECK: %{{.+}} = stream.timepoint.await %[[IF]]#2 => %[[IF]]#1
  %final2 = stream.timepoint.await %if_tp => %if_r2 : !stream.resource<external>{%c8}
  util.return %final1, %final2 : !stream.resource<external>, !stream.resource<external>
}

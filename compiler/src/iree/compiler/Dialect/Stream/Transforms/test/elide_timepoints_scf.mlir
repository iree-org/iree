// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-stream-schedule-execution),iree-stream-elide-timepoints)" %s | FileCheck %s

// Tests that await is sunk into the else branch when only the else branch uses
// the value directly (yields it without additional operations).

// CHECK-LABEL: @scfIfSinkAwaitIntoElse
util.func public @scfIfSinkAwaitIntoElse(%cond: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0_i32 = arith.constant 0 : i32
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // CHECK: %[[RESULTS:.+]], %[[TP:.+]] = stream.async.execute
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}

  // CHECK-NOT: stream.timepoint.await
  // CHECK: %{{.+}} = scf.if
  %result = scf.if %cond -> !stream.resource<external> {
    // Then: independent computation, doesn't use clone.
    // CHECK: stream.async.execute
    // CHECK-NEXT: stream.async.splat
    %other = stream.async.splat %c0_i32 : i32 -> !stream.resource<external>{%c4}
    scf.yield %other : !stream.resource<external>
  } else {
    // Else: yields clone directly - await should be sunk here.
    // CHECK: %[[READY:.+]] = stream.timepoint.await %[[TP]] => %[[RESULTS]]
    // CHECK-NEXT: scf.yield %[[READY]]
    scf.yield %clone : !stream.resource<external>
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

  // CHECK: %[[RESULTS:.+]], %[[TP:.+]] = stream.async.execute
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}

  // CHECK-NOT: stream.timepoint.await
  // CHECK: %{{.+}} = scf.if
  %result = scf.if %cond -> !stream.resource<external> {
    // Then: yields clone directly - await should be sunk here.
    // CHECK: %[[READY:.+]] = stream.timepoint.await %[[TP]] => %[[RESULTS]]
    // CHECK-NEXT: scf.yield %[[READY]]
    scf.yield %clone : !stream.resource<external>
  } else {
    // Else: independent computation.
    // CHECK: stream.async.execute
    // CHECK-NEXT: stream.async.splat
    %other = stream.async.splat %c0_i32 : i32 -> !stream.resource<external>{%c4}
    scf.yield %other : !stream.resource<external>
  }

  util.return %result : !stream.resource<external>
}

// -----

// Tests that await is sunk into BOTH branches when both branches use the value
// directly but perform different operations (in mutually exclusive execution
// paths).

// CHECK-LABEL: @scfIfSinkIntoBothBranches
util.func public @scfIfSinkIntoBothBranches(%cond: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // CHECK: %[[RESULTS:.+]], %[[TP:.+]] = stream.async.execute
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}

  // Both branches use clone but perform different slices.
  // CHECK-NOT: stream.timepoint.await %[[TP]]
  // CHECK: %{{.+}} = scf.if %{{.+}} -> (!stream.resource<external>)
  %result = scf.if %cond -> !stream.resource<external> {
    // Then: uses clone for first half slice.
    // CHECK: stream.async.execute await(%[[TP]])
    // CHECK-SAME: with(%[[RESULTS]] as %{{.+}}: !stream.resource<external>
    // CHECK: stream.async.slice
    %slice0 = stream.async.slice %clone[%c0 to %c4] : !stream.resource<external>{%c8} -> !stream.resource<external>{%c4}
    scf.yield %slice0 : !stream.resource<external>
  } else {
    // Else: uses clone for second half slice.
    // CHECK: stream.async.execute await(%[[TP]])
    // CHECK-SAME: with(%[[RESULTS]] as %{{.+}}: !stream.resource<external>
    // CHECK: stream.async.slice
    %slice1 = stream.async.slice %clone[%c4 to %c8] : !stream.resource<external>{%c8} -> !stream.resource<external>{%c4}
    scf.yield %slice1 : !stream.resource<external>
  }

  util.return %result : !stream.resource<external>
}

// -----

// Tests that await is hoisted past scf.if when the value is not used in any
// branch, only after the control flow.

// CHECK-LABEL: @scfIfHoistAwaitAfter
util.func public @scfIfHoistAwaitAfter(%cond: i1, %arg0: !stream.resource<external>) -> (!stream.resource<external>, !stream.resource<external>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // CHECK: %[[RESULTS:.+]], %[[TP:.+]] = stream.async.execute
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}

  // scf.if doesn't use clone at all - produces independent resources.
  // CHECK-NOT: stream.timepoint.await
  // CHECK: %[[BRANCH:.+]] = scf.if
  %branch_result = scf.if %cond -> !stream.resource<external> {
    // CHECK: stream.async.execute
    // CHECK-NEXT: stream.async.splat
    %splat0 = stream.async.splat %c0_i32 : i32 -> !stream.resource<external>{%c4}
    scf.yield %splat0 : !stream.resource<external>
  } else {
    // CHECK: stream.async.execute
    // CHECK-NEXT: stream.async.splat
    %splat1 = stream.async.splat %c1_i32 : i32 -> !stream.resource<external>{%c4}
    scf.yield %splat1 : !stream.resource<external>
  }

  // clone is used AFTER the control flow - await can stay here (not blocking).
  // Since clone isn't used in any branch, the await doesn't need to move.
  // CHECK: %[[READY:.+]] = stream.timepoint.await %[[TP]] => %[[RESULTS]]
  util.return %clone, %branch_result : !stream.resource<external>, !stream.resource<external>
}

// -----

// Tests that await followed immediately by execute is folded into
// execute await() clause.

// CHECK-LABEL: @scfIfFoldAwaitWithExecute
util.func public @scfIfFoldAwaitWithExecute(%cond: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index

  // CHECK: %[[RESULTS:.+]], %[[TP:.+]] = stream.async.execute
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c16} -> !stream.resource<external>{%c16}

  // scf.if produces a resource (not just an index).
  // CHECK: %{{.+}} = scf.if %{{.+}} -> (!stream.resource<external>)
  %branch_resource = scf.if %cond -> !stream.resource<external> {
    // CHECK: stream.async.execute
    // CHECK: stream.async.splat
    %splat0 = stream.async.splat %c0_i32 : i32 -> !stream.resource<external>{%c16}
    scf.yield %splat0 : !stream.resource<external>
  } else {
    // CHECK: stream.async.execute
    // CHECK: stream.async.splat
    %splat1 = stream.async.splat %c1_i32 : i32 -> !stream.resource<external>{%c16}
    scf.yield %splat1 : !stream.resource<external>
  }

  // After control flow we use both the clone and branch resource.
  // Copy 4 bytes from clone to branch_resource at offset 4-8.
  // The await + execute should be folded.
  // CHECK: stream.async.execute await(%[[TP]]) =>
  // CHECK-SAME: with(
  // CHECK: stream.async.copy
  %result = stream.async.copy %clone[%c0 to %c4], %branch_resource[%c4 to %c8], %c4 : !stream.resource<external>{%c16} -> %branch_resource as !stream.resource<external>{%c16}

  util.return %result : !stream.resource<external>
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

  // CHECK: %[[RESULTS:.+]], %[[TP:.+]] = stream.async.execute
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c4} -> !stream.resource<external>{%c4}

  // Conservative: no sinking into loop - the clone timepoint is captured by
  // stream.async.execute in the loop body via await().
  // CHECK-NOT: stream.timepoint.await %[[TP]]
  // CHECK: scf.for
  %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%iter = %c0_i32) -> (i32) {
    // Loop body uses clone in stream.async.execute await clause.
    // CHECK: stream.async.execute await(%[[TP]])
    %transfer = stream.async.transfer %clone : !stream.resource<external>{%c4} -> !stream.resource<staging>{%c4}
    %load = stream.async.load %transfer[%c0] : !stream.resource<staging>{%c4} -> i32
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

  // CHECK: %[[RESULTS:.+]], %[[TP:.+]] = stream.async.execute
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c4} -> !stream.resource<external>{%c4}

  // Conservative: no sinking into while - the clone timepoint is captured by
  // stream.async.execute in the while body via await().
  // CHECK-NOT: stream.timepoint.await %[[TP]]
  // CHECK: scf.while
  %result:2 = scf.while (%iter = %c0_i32, %count = %c0) : (i32, index) -> (i32, index) {
    %cond = arith.cmpi slt, %count, %c10 : index
    scf.condition(%cond) %iter, %count : i32, index
  } do {
  ^bb0(%iter: i32, %count: index):
    // While body uses clone in stream.async.execute await clause.
    // CHECK: stream.async.execute await(%[[TP]])
    %transfer = stream.async.transfer %clone : !stream.resource<external>{%c4} -> !stream.resource<staging>{%c4}
    %load = stream.async.load %transfer[%c0] : !stream.resource<staging>{%c4} -> i32
    %add = arith.addi %iter, %load : i32
    %next_count = arith.addi %count, %c1 : index
    scf.yield %add, %next_count : i32, index
  }

  util.return %result#0 : i32
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

  // CHECK: %[[RESULTS:.+]], %[[TP:.+]] = stream.async.execute
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}

  // Outer scf.if: clone used only in else branch.
  // CHECK-NOT: stream.timepoint.await
  // CHECK: %{{.+}} = scf.if %{{.+}} -> (!stream.resource<external>)
  %outer_result = scf.if %outer_cond -> !stream.resource<external> {
    // Then branch: independent, has nested scf.if.
    // CHECK: %{{.+}} = scf.if
    %inner_result = scf.if %inner_cond -> !stream.resource<external> {
      // CHECK: stream.async.execute
      // CHECK-NEXT: stream.async.splat
      %splat0 = stream.async.splat %c0_i32 : i32 -> !stream.resource<external>{%c4}
      scf.yield %splat0 : !stream.resource<external>
    } else {
      // CHECK: stream.async.execute
      // CHECK-NEXT: stream.async.splat
      %splat1 = stream.async.splat %c4_i32 : i32 -> !stream.resource<external>{%c4}
      scf.yield %splat1 : !stream.resource<external>
    }
    scf.yield %inner_result : !stream.resource<external>
  } else {
    // Else branch: yields clone directly - await should be sunk here.
    // CHECK: %[[READY:.+]] = stream.timepoint.await %[[TP]] => %[[RESULTS]]
    // CHECK-NEXT: scf.yield %[[READY]]
    scf.yield %clone : !stream.resource<external>
  }

  util.return %outer_result : !stream.resource<external>
}
// -----

// Tests scf.index_switch with multiple cases where different cases use the
// async value. Should sink into each case that uses it.

// CHECK-LABEL: @scfIndexSwitchMultipleCases
util.func public @scfIndexSwitchMultipleCases(%selector: index, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // CHECK: %[[RESULTS:.+]], %[[TP:.+]] = stream.async.execute
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}

  // scf.index_switch: case 0 and 2 use clone, case 1 and default don't.
  // CHECK-NOT: stream.timepoint.await %[[TP]]
  // CHECK: %{{.+}} = scf.index_switch %{{.+}} -> !stream.resource<external>
  %result = scf.index_switch %selector -> !stream.resource<external>
  case 0 {
    // Case 0: uses clone via execute with await.
    // CHECK: stream.async.execute await(%[[TP]])
    // CHECK-SAME: with(%[[RESULTS]] as %{{.+}}: !stream.resource<external>
    // CHECK: stream.async.slice
    %slice0 = stream.async.slice %clone[%c0 to %c4] : !stream.resource<external>{%c8} -> !stream.resource<external>{%c4}
    scf.yield %slice0 : !stream.resource<external>
  }
  case 1 {
    // Case 1: independent computation, doesn't use clone.
    // CHECK: stream.async.execute
    // CHECK-NEXT: stream.async.splat
    %splat1 = stream.async.splat %c1_i32 : i32 -> !stream.resource<external>{%c4}
    scf.yield %splat1 : !stream.resource<external>
  }
  case 2 {
    // Case 2: uses clone via execute with await.
    // CHECK: stream.async.execute await(%[[TP]])
    // CHECK-SAME: with(%[[RESULTS]] as %{{.+}}: !stream.resource<external>
    // CHECK: stream.async.slice
    %slice2 = stream.async.slice %clone[%c4 to %c8] : !stream.resource<external>{%c8} -> !stream.resource<external>{%c4}
    scf.yield %slice2 : !stream.resource<external>
  }
  default {
    // Default: independent computation.
    // CHECK: stream.async.execute
    // CHECK-NEXT: stream.async.splat
    %splat_default = stream.async.splat %c2_i32 : i32 -> !stream.resource<external>{%c4}
    scf.yield %splat_default : !stream.resource<external>
  }

  util.return %result : !stream.resource<external>
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

  // CHECK: %[[RESULTS:.+]], %[[TP:.+]] = stream.async.execute
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}

  // Outer scf.if: clone used only in else branch's nested scf.if.
  // CHECK-NOT: stream.timepoint.await %[[TP]]
  // CHECK: %{{.+}} = scf.if %{{.+}} -> (!stream.resource<external>)
  %outer_result = scf.if %outer -> !stream.resource<external> {
    // Outer then: independent.
    // CHECK: stream.async.execute
    // CHECK-NEXT: stream.async.splat
    %splat = stream.async.splat %c0_i32 : i32 -> !stream.resource<external>{%c4}
    scf.yield %splat : !stream.resource<external>
  } else {
    // Outer else: has middle scf.if.
    // CHECK: %{{.+}} = scf.if
    %middle_result = scf.if %middle -> !stream.resource<external> {
      // Middle then: independent.
      // CHECK: stream.async.execute
      // CHECK-NEXT: stream.async.splat
      %splat_mid = stream.async.splat %c1_i32 : i32 -> !stream.resource<external>{%c4}
      scf.yield %splat_mid : !stream.resource<external>
    } else {
      // Middle else: has inner scf.if where clone is used.
      // CHECK: %{{.+}} = scf.if
      %inner_result = scf.if %inner -> !stream.resource<external> {
        // Inner then: uses clone - await sunk to innermost level.
        // CHECK: stream.async.execute await(%[[TP]])
        // CHECK-SAME: with(%[[RESULTS]] as %{{.+}}: !stream.resource<external>
        // CHECK: stream.async.slice
        %slice = stream.async.slice %clone[%c0 to %c4] : !stream.resource<external>{%c8} -> !stream.resource<external>{%c4}
        scf.yield %slice : !stream.resource<external>
      } else {
        // Inner else: yields clone directly - await sunk here too.
        // CHECK: %[[READY:.+]] = stream.timepoint.await %[[TP]] => %[[RESULTS]]
        // CHECK-NEXT: scf.yield %[[READY]]
        scf.yield %clone : !stream.resource<external>
      }
      scf.yield %inner_result : !stream.resource<external>
    }
    scf.yield %middle_result : !stream.resource<external>
  }

  util.return %outer_result : !stream.resource<external>
}

// -----

// Tests that await is correctly handled when positioned between nested
// control flow operations.

// CHECK-LABEL: @scfIfAwaitBetweenNestedOps
util.func public @scfIfAwaitBetweenNestedOps(%cond1: i1, %cond2: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // First async operation.
  // CHECK: %[[RESULTS1:.+]], %[[TP1:.+]] = stream.async.execute
  %clone1 = stream.async.clone %arg0 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}

  // First scf.if uses clone1.
  // CHECK-NOT: stream.timepoint.await %[[TP1]]
  // CHECK: %[[IF1:.+]] = scf.if
  %result1 = scf.if %cond1 -> !stream.resource<external> {
    // CHECK: stream.async.execute await(%[[TP1]])
    // CHECK-SAME: with(%[[RESULTS1]] as %{{.+}}: !stream.resource<external>
    %slice1 = stream.async.slice %clone1[%c0 to %c4] : !stream.resource<external>{%c8} -> !stream.resource<external>{%c4}
    scf.yield %slice1 : !stream.resource<external>
  } else {
    // CHECK: stream.async.execute
    // CHECK-NEXT: stream.async.splat
    %splat1 = stream.async.splat %c0_i32 : i32 -> !stream.resource<external>{%c4}
    scf.yield %splat1 : !stream.resource<external>
  }

  // Second async operation between the two scf.if ops.
  // CHECK: %[[RESULTS2:.+]], %[[TP2:.+]] = stream.async.execute
  %clone2 = stream.async.clone %result1 : !stream.resource<external>{%c4} -> !stream.resource<external>{%c4}

  // Second scf.if uses clone2.
  // CHECK-NOT: stream.timepoint.await %[[TP2]]
  // CHECK: %{{.+}} = scf.if
  %result2 = scf.if %cond2 -> !stream.resource<external> {
    // CHECK: %[[READY:.+]] = stream.timepoint.await %[[TP2]] => %[[RESULTS2]]
    // CHECK-NEXT: scf.yield %[[READY]]
    scf.yield %clone2 : !stream.resource<external>
  } else {
    // CHECK: stream.async.execute
    // CHECK-NEXT: stream.async.splat
    %splat2 = stream.async.splat %c1_i32 : i32 -> !stream.resource<external>{%c4}
    scf.yield %splat2 : !stream.resource<external>
  }

  util.return %result2 : !stream.resource<external>
}

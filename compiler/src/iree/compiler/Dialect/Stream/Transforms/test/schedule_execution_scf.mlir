// RUN: iree-opt --split-input-file --allow-unregistered-dialect --pass-pipeline="builtin.module(util.func(iree-stream-schedule-execution))" %s | FileCheck %s

// Tests basic scf.for loop with dependency on cloned resource (reproducer case).
// The loop uses a cloned resource from a partition, which creates a dependency
// that must be respected during partitioning.

// CHECK-LABEL: @scfForWithDependency
util.func public @scfForWithDependency(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c11 = arith.constant 11 : index
  %c48 = arith.constant 48 : index
  %c0_i64 = arith.constant 0 : i64
  %c5_i64 = arith.constant 5 : i64

  // CHECK: scf.for
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c48} -> !stream.resource<external>{%c48}


  %result = scf.for %i = %c0 to %c11 step %c1 iter_args(%iter = %c0_i64) -> (i64) {
    %offset = arith.muli %i, %c4 : index
    %end = arith.addi %offset, %c4 : index
    // Loop uses the cloned resource - this should NOT be partitioned into the clone partition.
    // CHECK: %[[EXEC_RESULT:.+]], %[[EXEC_TIMEPOINT:.+]] = stream.async.execute
    // CHECK-SAME: with(%{{.+}} as %[[CAPTURE:.+]]: !stream.resource<external>{%c48})
    // CHECK-SAME: -> !stream.resource<staging>{%c4}
    // CHECK-NEXT: %[[CLONE:.+]] = stream.async.clone %[[CAPTURE]]
    // CHECK-NEXT: %[[SLICE:.+]] = stream.async.slice %[[CLONE]]
    // CHECK-NEXT: %[[TRANSFER:.+]] = stream.async.transfer %[[SLICE]]
    // CHECK-NEXT: stream.yield %[[TRANSFER]]
    %slice = stream.async.slice %clone[%offset to %end] : !stream.resource<external>{%c48} -> !stream.resource<external>{%c4}
    %transfer = stream.async.transfer %slice : !stream.resource<external>{%c4} -> !stream.resource<staging>{%c4}
    // CHECK: stream.timepoint.await %[[EXEC_TIMEPOINT]]
    %load = stream.async.load %transfer[%c0] : !stream.resource<staging>{%c4} -> i32
    %ext = arith.extsi %load : i32 to i64
    %cmp = arith.cmpi eq, %ext, %c5_i64 : i64
    %add = arith.extui %cmp : i1 to i64
    %next = arith.addi %iter, %add : i64
    scf.yield %next : i64
  }

  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.splat
  %splat = stream.async.splat %result : i64 -> !stream.resource<external>{%c4}
  util.return %splat : !stream.resource<external>
}

// -----

// Tests scf.for loop result being used by a dispatch.
// The loop produces a value that is consumed by a partition.

stream.async.func private @dispatch(%arg0: i64, %arg1: !stream.resource<*>) -> !stream.resource<*>

// CHECK-LABEL: @scfForProducingDispatchInput
util.func public @scfForProducingDispatchInput(%arg0: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : index
  %c128 = arith.constant 128 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64

  // CHECK: scf.for
  %count = scf.for %i = %c0 to %c10 step %c1 iter_args(%iter = %c0_i64) -> (i64) {
    %next = arith.addi %iter, %c1_i64 : i64
    scf.yield %next : i64
  }

  // The dispatch should be in a partition that waits for the loop to complete.
  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.call @dispatch
  %result = stream.async.call @dispatch(%count, %arg0[%c0 to %c128 for %c128]) : (i64, !stream.resource<*>{%c128}) -> %arg0{%c128}
  // CHECK: stream.timepoint.await
  util.return %result : !stream.resource<*>
}

// -----

// Tests nested scf.for loops with stream operations.
// Both inner and outer loops should remain outside partitions.

// CHECK-LABEL: @scfNestedFor
util.func public @scfNestedFor(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c0_i32 = arith.constant 0 : i32

  // CHECK: scf.for
  %outer_result = scf.for %i = %c0 to %c3 step %c1 iter_args(%outer_iter = %c0_i32) -> (i32) {
    // CHECK: scf.for
    %inner_result = scf.for %j = %c0 to %c3 step %c1 iter_args(%inner_iter = %outer_iter) -> (i32) {
      %offset = arith.muli %j, %c4 : index
      %end = arith.addi %offset, %c4 : index
      // CHECK: stream.async.execute
      // CHECK-NEXT: stream.async.slice
      // CHECK-NEXT: stream.async.transfer
      %slice = stream.async.slice %arg0[%offset to %end] : !stream.resource<external>{%c16} -> !stream.resource<external>{%c4}
      %transfer = stream.async.transfer %slice : !stream.resource<external>{%c4} -> !stream.resource<staging>{%c4}
      // CHECK: stream.timepoint.await
      %load = stream.async.load %transfer[%c0] : !stream.resource<staging>{%c4} -> i32
      %add = arith.addi %inner_iter, %load : i32
      scf.yield %add : i32
    }
    scf.yield %inner_result : i32
  }

  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.splat
  %splat = stream.async.splat %outer_result : i32 -> !stream.resource<external>{%c4}
  util.return %splat : !stream.resource<external>
}

// -----

// Tests scf.for loop that uses multiple values produced by a single partition
// before the loop. The key expectations:
// 1. The two slices are created in a single execute region before the loop.
// 2. The loop does NOT await the timepoint before entering.
// 3. Inside the loop body, both slices are used in a single execute region
//    that awaits the partition's timepoint.
// 4. The two transfers are merged into one execute region.

// CHECK-LABEL: @scfForWithMultipleRegionUses
util.func public @scfForWithMultipleRegionUses(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c0_i32 = arith.constant 0 : i32

  // CHECK: %[[RESULTS:.+]]:2, %[[TIMEPOINT:.+]] = stream.async.execute
  // CHECK-SAME: with(%{{.+}} as %[[CAPTURE:.+]]: !stream.resource<external>{%c16})
  // CHECK-SAME: -> (!stream.resource<external>{%c8}, !stream.resource<external>{%c8})
  // CHECK-NEXT: %[[SLICE0:.+]] = stream.async.slice %[[CAPTURE]][%c0 to %c8]
  %slice0 = stream.async.slice %arg0[%c0 to %c8] : !stream.resource<external>{%c16} -> !stream.resource<external>{%c8}
  // CHECK-NEXT: %[[SLICE1:.+]] = stream.async.slice %[[CAPTURE]][%c8 to %c16]
  %slice1 = stream.async.slice %arg0[%c8 to %c16] : !stream.resource<external>{%c16} -> !stream.resource<external>{%c8}
  // CHECK-NEXT: stream.yield %[[SLICE0]], %[[SLICE1]]

  // CHECK-NOT: stream.timepoint.await
  // CHECK: scf.for
  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%iter = %c0_i32) -> (i32) {
    // Loop uses both slices from the partition. Both transfers merged into one execute.
    // CHECK: stream.async.execute await(%[[TIMEPOINT]])
    // CHECK-NEXT: stream.async.transfer
    %transfer0 = stream.async.transfer %slice0 : !stream.resource<external>{%c8} -> !stream.resource<staging>{%c8}
    // CHECK-NEXT: stream.async.transfer
    %transfer1 = stream.async.transfer %slice1 : !stream.resource<external>{%c8} -> !stream.resource<staging>{%c8}
    // CHECK: stream.timepoint.await
    %load0 = stream.async.load %transfer0[%c0] : !stream.resource<staging>{%c8} -> i32

    // CHECK-NOT: stream.async.execute
    // CHECK-NOT: stream.timepoint.await
    %load1 = stream.async.load %transfer1[%c0] : !stream.resource<staging>{%c8} -> i32

    %add0 = arith.addi %iter, %load0 : i32
    %add1 = arith.addi %add0, %load1 : i32
    scf.yield %add1 : i32
  }

  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.splat
  %splat = stream.async.splat %result : i32 -> !stream.resource<external>{%c4}
  util.return %splat : !stream.resource<external>
}

// -----

// Tests scf.for loop with zero iterations.
// The loop may not execute at all, which affects partitioning.

// CHECK-LABEL: @scfForZeroIterations
util.func public @scfForZeroIterations(%arg0: !stream.resource<external>, %lb: index, %ub: index) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c0_i32 = arith.constant 0 : i32

  // Clone is NOT materialized outside the loop since the loop may not execute.
  // CHECK-NOT: stream.async.execute
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c16} -> !stream.resource<external>{%c16}

  // CHECK: scf.for
  %result = scf.for %i = %lb to %ub step %c1 iter_args(%iter = %c0_i32) -> (i32) {
    // Clone is materialized INSIDE the execute region within the loop body.
    // CHECK: stream.async.execute
    // CHECK-NEXT: stream.async.clone
    // CHECK-NEXT: stream.async.slice
    // CHECK-NEXT: stream.async.transfer
    %slice = stream.async.slice %clone[%c0 to %c4] : !stream.resource<external>{%c16} -> !stream.resource<external>{%c4}
    %transfer = stream.async.transfer %slice : !stream.resource<external>{%c4} -> !stream.resource<staging>{%c4}
    // CHECK: stream.timepoint.await
    %load = stream.async.load %transfer[%c0] : !stream.resource<staging>{%c4} -> i32
    %add = arith.addi %iter, %load : i32
    scf.yield %add : i32
  }

  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.splat
  %splat = stream.async.splat %result : i32 -> !stream.resource<external>{%c4}
  util.return %splat : !stream.resource<external>
}

// -----

// Tests scf.if with both then and else branches containing stream operations.
// Both branches should be partitioned independently.

// CHECK-LABEL: @scfIfThenElse
util.func public @scfIfThenElse(%cond: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index

  // CHECK: scf.if
  %result = scf.if %cond -> !stream.resource<external> {
    // CHECK: %[[RESULTS_0:.+]], %[[RESULT_TIMEPOINT_1:.+]] = stream.async.execute
    // CHECK-SAME: with(%{{.+}} as %[[ARG2:.+]]: !stream.resource<external>{%c16})
    // CHECK-SAME: -> !stream.resource<external>{%c8}
    // CHECK-NEXT: %[[SLICE:.+]] = stream.async.slice %[[ARG2]][%c0 to %c8]
    %slice = stream.async.slice %arg0[%c0 to %c8] : !stream.resource<external>{%c16} -> !stream.resource<external>{%c8}
    // CHECK-NEXT: stream.yield %[[SLICE]]
    // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[RESULT_TIMEPOINT_1]] => %[[RESULTS_0]]
    // CHECK: scf.yield %[[AWAITED]]
    scf.yield %slice : !stream.resource<external>
  } else {
    // CHECK: %[[RESULTS_0:.+]], %[[RESULT_TIMEPOINT_1:.+]] = stream.async.execute
    // CHECK-SAME: with(%{{.+}} as %[[ARG2:.+]]: !stream.resource<external>{%c16})
    // CHECK-SAME: -> !stream.resource<external>{%c8}
    // CHECK-NEXT: %[[SLICE:.+]] = stream.async.slice %[[ARG2]][%c8 to %c16]
    %slice = stream.async.slice %arg0[%c8 to %c16] : !stream.resource<external>{%c16} -> !stream.resource<external>{%c8}
    // CHECK-NEXT: stream.yield %[[SLICE]]
    // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[RESULT_TIMEPOINT_1]] => %[[RESULTS_0]]
    // CHECK: scf.yield %[[AWAITED]]
    scf.yield %slice : !stream.resource<external>
  }

  // CHECK: %[[RESULTS:.+]], %[[RESULT_TIMEPOINT:.+]] = stream.async.execute
  // CHECK-SAME: with(%{{.+}} as %[[ARG2:.+]]: !stream.resource<external>{%c8})
  // CHECK-SAME: -> !stream.resource<external>{%c8}
  // CHECK-NEXT: %[[CLONE:.+]] = stream.async.clone %[[ARG2]]
  %transfer = stream.async.transfer %result : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}
  // CHECK-NEXT: stream.yield %[[CLONE]]
  // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[RESULT_TIMEPOINT]] => %[[RESULTS]]
  // CHECK: util.return %[[AWAITED]]
  util.return %transfer : !stream.resource<external>
}

// -----

// Tests scf.if with only then branch (no else).
// The then branch contains stream operations.

// CHECK-LABEL: @scfIfThenOnly
util.func public @scfIfThenOnly(%cond: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // CHECK: scf.if
  %result = scf.if %cond -> !stream.resource<external> {
    // CHECK: %[[RESULTS:.+]], %[[RESULT_TIMEPOINT:.+]] = stream.async.execute
    // CHECK-SAME: with(%{{.+}} as %[[ARG2:.+]]: !stream.resource<external>{%c8})
    // CHECK-SAME: -> !stream.resource<external>{%c4}
    // CHECK-NEXT: %[[SLICE:.+]] = stream.async.slice %[[ARG2]][%c0 to %c4]
    %slice = stream.async.slice %arg0[%c0 to %c4] : !stream.resource<external>{%c8} -> !stream.resource<external>{%c4}
    // CHECK-NEXT: stream.yield %[[SLICE]]
    // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[RESULT_TIMEPOINT]] => %[[RESULTS]]
    // CHECK: scf.yield %[[AWAITED]]
    scf.yield %slice : !stream.resource<external>
  } else {
    scf.yield %arg0 : !stream.resource<external>
  }

  // CHECK: util.return
  util.return %result : !stream.resource<external>
}

// -----

// Tests scf.if using cloned resource from partition.
// The if operation depends on a partition output.

// CHECK-LABEL: @scfIfWithDependency
util.func public @scfIfWithDependency(%cond: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // CHECK: %[[RESULTS:.+]], %[[RESULT_TIMEPOINT:.+]] = stream.async.execute
  // CHECK-SAME: with(%{{.+}} as %[[ARG2:.+]]: !stream.resource<external>{%c8})
  // CHECK-SAME: -> !stream.resource<external>{%c8}
  // CHECK-NEXT: %[[CLONE_OP:.+]] = stream.async.clone %[[ARG2]]
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}
  // CHECK-NEXT: stream.yield %[[CLONE_OP]]

  // The else branch uses clone directly (not in a stream op), so await must happen before scf.if.
  // CHECK: %[[CLONE_READY:.+]] = stream.timepoint.await %[[RESULT_TIMEPOINT]] => %[[RESULTS]]
  // CHECK: scf.if
  %result = scf.if %cond -> !stream.resource<external> {
    // Then branch uses cloned resource in stream op - can reference original clone.
    // CHECK: %[[RESULTS_0:.+]], %[[RESULT_TIMEPOINT_1:.+]] = stream.async.execute await(%[[RESULT_TIMEPOINT]])
    // CHECK-SAME: with(%[[RESULTS]] as %[[ARG2:.+]]: !stream.resource<external>{%c8})
    // CHECK-SAME: -> !stream.resource<external>{%c4}
    // CHECK-NEXT: %[[SLICE:.+]] = stream.async.slice %[[ARG2]][%c0 to %c4]
    %slice = stream.async.slice %clone[%c0 to %c4] : !stream.resource<external>{%c8} -> !stream.resource<external>{%c4}
    // CHECK-NEXT: stream.yield %[[SLICE]]
    // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[RESULT_TIMEPOINT_1]] => %[[RESULTS_0]]
    // CHECK: scf.yield %[[AWAITED]]
    scf.yield %slice : !stream.resource<external>
  } else {
    // CHECK: scf.yield %[[CLONE_READY]]
    scf.yield %clone : !stream.resource<external>
  }

  // CHECK: util.return
  util.return %result : !stream.resource<external>
}

// -----

// Tests scf.if result being used by a dispatch.
// The if produces a value consumed by a partition.

stream.async.func private @dispatch(%arg0: !stream.resource<*>, %arg1: index) -> %arg0

// CHECK-LABEL: @scfIfProducingDispatchInput
util.func public @scfIfProducingDispatchInput(%cond: i1, %arg0: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index

  %size = scf.if %cond -> index {
    // To prevent folding to arith.select.
    "some.sideeffect"() : () -> ()
    scf.yield %c128 : index
  } else {
    scf.yield %c256 : index
  }

  // The dispatch should be in a partition.
  // CHECK: %[[RESULTS:.+]], %[[RESULT_TIMEPOINT:.+]] = stream.async.execute
  // CHECK-NEXT: stream.async.call @dispatch
  %result = stream.async.call @dispatch(%arg0[%c0 to %size for %size], %size) : (!stream.resource<*>{%size}, index) -> %arg0{%size}
  // CHECK: stream.timepoint.await %[[RESULT_TIMEPOINT]] => %[[RESULTS]]
  util.return %result : !stream.resource<*>
}

// -----

// Tests scf.if nested inside scf.for with stream operations.
// Both control flow operations should remain outside partitions.

// CHECK-LABEL: @scfNestedIfInFor
util.func public @scfNestedIfInFor(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c0_i32 = arith.constant 0 : i32

  // CHECK: scf.for
  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%iter = %c0_i32) -> (i32) {
    %c2 = arith.constant 2 : index
    %rem = arith.remui %i, %c2 : index
    %cond = arith.cmpi eq, %rem, %c0 : index

    // CHECK: scf.if
    %value = scf.if %cond -> i32 {
      // CHECK: %[[RESULTS_0:.+]], %[[RESULT_TIMEPOINT_1:.+]] = stream.async.execute
      // CHECK-SAME: with(%{{.+}} as %[[ARG3:.+]]: !stream.resource<external>{%c16})
      // CHECK-SAME: -> !stream.resource<staging>{%c4}
      // CHECK-NEXT: %[[SLICE:.+]] = stream.async.slice %[[ARG3]][%c0 to %c4]
      // CHECK-NEXT: %[[TRANSFER:.+]] = stream.async.transfer %[[SLICE]]
      %slice = stream.async.slice %arg0[%c0 to %c4] : !stream.resource<external>{%c16} -> !stream.resource<external>{%c4}
      %transfer = stream.async.transfer %slice : !stream.resource<external>{%c4} -> !stream.resource<staging>{%c4}
      // CHECK-NEXT: stream.yield %[[TRANSFER]]
      // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[RESULT_TIMEPOINT_1]] => %[[RESULTS_0]]
      // CHECK: %[[LOAD:.+]] = stream.async.load %[[AWAITED]][%c0]
      %load = stream.async.load %transfer[%c0] : !stream.resource<staging>{%c4} -> i32
      // CHECK: scf.yield %[[LOAD]]
      scf.yield %load : i32
    } else {
      // CHECK: %[[RESULTS_0:.+]], %[[RESULT_TIMEPOINT_1:.+]] = stream.async.execute
      // CHECK-SAME: with(%{{.+}} as %[[ARG3:.+]]: !stream.resource<external>{%c16})
      // CHECK-SAME: -> !stream.resource<staging>{%c8}
      // CHECK-NEXT: %[[SLICE:.+]] = stream.async.slice %[[ARG3]][%c8 to %c16]
      // CHECK-NEXT: %[[TRANSFER:.+]] = stream.async.transfer %[[SLICE]]
      %slice = stream.async.slice %arg0[%c8 to %c16] : !stream.resource<external>{%c16} -> !stream.resource<external>{%c8}
      %transfer = stream.async.transfer %slice : !stream.resource<external>{%c8} -> !stream.resource<staging>{%c8}
      // CHECK-NEXT: stream.yield %[[TRANSFER]]
      // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[RESULT_TIMEPOINT_1]] => %[[RESULTS_0]]
      // CHECK: %[[LOAD:.+]] = stream.async.load %[[AWAITED]][%c0]
      %load = stream.async.load %transfer[%c0] : !stream.resource<staging>{%c8} -> i32
      // CHECK: scf.yield %[[LOAD]]
      scf.yield %load : i32
    }

    %add = arith.addi %iter, %value : i32
    scf.yield %add : i32
  }

  // CHECK: %[[RESULTS:.+]], %[[RESULT_TIMEPOINT:.+]] = stream.async.execute
  // CHECK-NEXT: stream.async.splat
  %splat = stream.async.splat %result : i32 -> !stream.resource<external>{%c4}
  // CHECK: stream.timepoint.await %[[RESULT_TIMEPOINT]] => %[[RESULTS]]
  util.return %splat : !stream.resource<external>
}

// -----

// Tests basic scf.while loop with stream operations in before/after regions.
// Both regions should be partitioned independently.

// CHECK-LABEL: @scfWhileBasic
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<external>)
util.func public @scfWhileBasic(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c10 = arith.constant 10 : index
  %c0_i32 = arith.constant 0 : i32
  %c100_i32 = arith.constant 100 : i32

  // CHECK: %[[WHILE_RESULTS:.+]]:2 = scf.while
  // CHECK-SAME: (%[[ARG1:.+]] = %c0_i32, %[[ARG2:.+]] = %c0)
  // CHECK-SAME: (i32, index) -> (i32, index)
  %result:2 = scf.while (%iter = %c0_i32, %count = %c0) : (i32, index) -> (i32, index) {
    %cond = arith.cmpi slt, %count, %c10 : index
    // CHECK: scf.condition
    scf.condition(%cond) %iter, %count : i32, index
  } do {
  ^bb0(%iter: i32, %count: index):
    // CHECK: ^bb0(%[[ITER:.+]]: i32, %[[COUNT:.+]]: index):
    // CHECK: %[[RESULTS:.+]], %[[TIMEPOINT:.+]] = stream.async.execute
    // CHECK-SAME: with(%[[ARG0]] as %[[CAPTURE:.+]]: !stream.resource<external>{%c8})
    // CHECK-SAME: -> !stream.resource<staging>{%c4}
    // CHECK-NEXT: %[[SLICE:.+]] = stream.async.slice %[[CAPTURE]][%c0 to %c4]
    // CHECK-NEXT: %[[TRANSFER:.+]] = stream.async.transfer %[[SLICE]]
    // CHECK-NEXT: stream.yield %[[TRANSFER]]
    %slice = stream.async.slice %arg0[%c0 to %c4] : !stream.resource<external>{%c8} -> !stream.resource<external>{%c4}
    %transfer = stream.async.transfer %slice : !stream.resource<external>{%c4} -> !stream.resource<staging>{%c4}
    // CHECK: %[[READY:.+]] = stream.timepoint.await %[[TIMEPOINT]] => %[[RESULTS]]
    %load = stream.async.load %transfer[%c0] : !stream.resource<staging>{%c4} -> i32
    // CHECK: %[[LOAD:.+]] = stream.async.load %[[READY]][%c0]
    %add = arith.addi %iter, %load : i32
    %next_count = arith.addi %count, %c1 : index
    scf.yield %add, %next_count : i32, index
  }

  // CHECK: %[[SPLAT_RESULTS:.+]], %[[SPLAT_TIMEPOINT:.+]] = stream.async.execute
  // CHECK-SAME: -> !stream.resource<external>{%c4}
  // CHECK-NEXT: %[[SPLAT:.+]] = stream.async.splat %[[WHILE_RESULTS]]#0
  // CHECK-NEXT: stream.yield %[[SPLAT]]
  %splat = stream.async.splat %result#0 : i32 -> !stream.resource<external>{%c4}
  // CHECK: %[[FINAL:.+]] = stream.timepoint.await %[[SPLAT_TIMEPOINT]] => %[[SPLAT_RESULTS]]
  // CHECK: util.return %[[FINAL]]
  util.return %splat : !stream.resource<external>
}

// -----

// Tests scf.while loop using cloned resource from partition.
// The while depends on a partition output.

// CHECK-LABEL: @scfWhileWithDependency
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<external>)
util.func public @scfWhileWithDependency(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c5 = arith.constant 5 : index
  %c0_i32 = arith.constant 0 : i32

  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}

  // CHECK: %[[WHILE_RESULTS:.+]]:2 = scf.while
  // CHECK-SAME: (%[[ARG1:.+]] = %c0_i32, %[[ARG2:.+]] = %c0)
  // CHECK-SAME: (i32, index) -> (i32, index)
  %result:2 = scf.while (%iter = %c0_i32, %count = %c0) : (i32, index) -> (i32, index) {
    %cond = arith.cmpi slt, %count, %c5 : index
    scf.condition(%cond) %iter, %count : i32, index
  } do {
  ^bb0(%iter: i32, %count: index):
    // CHECK: ^bb0(%[[ITER:.+]]: i32, %[[COUNT:.+]]: index):
    // While uses the cloned resource.
    // CHECK: %[[RESULTS:.+]], %[[TIMEPOINT:.+]] = stream.async.execute
    // CHECK-SAME: with(%[[ARG0]] as %[[CAPTURE:.+]]: !stream.resource<external>{%c8})
    // CHECK-SAME: -> !stream.resource<staging>{%c4}
    // CHECK-NEXT: %[[CLONE:.+]] = stream.async.clone %[[CAPTURE]]
    // CHECK-NEXT: %[[SLICE:.+]] = stream.async.slice %[[CLONE]][%c0 to %c4]
    // CHECK-NEXT: %[[TRANSFER:.+]] = stream.async.transfer %[[SLICE]]
    // CHECK-NEXT: stream.yield %[[TRANSFER]]
    %slice = stream.async.slice %clone[%c0 to %c4] : !stream.resource<external>{%c8} -> !stream.resource<external>{%c4}
    %transfer = stream.async.transfer %slice : !stream.resource<external>{%c4} -> !stream.resource<staging>{%c4}
    // CHECK: %[[READY:.+]] = stream.timepoint.await %[[TIMEPOINT]] => %[[RESULTS]]
    %load = stream.async.load %transfer[%c0] : !stream.resource<staging>{%c4} -> i32
    // CHECK: %[[LOAD:.+]] = stream.async.load %[[READY]][%c0]
    %add = arith.addi %iter, %load : i32
    %next_count = arith.addi %count, %c1 : index
    scf.yield %add, %next_count : i32, index
  }

  // CHECK: %[[SPLAT_RESULTS:.+]], %[[SPLAT_TIMEPOINT:.+]] = stream.async.execute
  // CHECK-SAME: -> !stream.resource<external>{%c4}
  // CHECK-NEXT: %[[SPLAT:.+]] = stream.async.splat %[[WHILE_RESULTS]]#0
  // CHECK-NEXT: stream.yield %[[SPLAT]]
  %splat = stream.async.splat %result#0 : i32 -> !stream.resource<external>{%c4}
  // CHECK: %[[FINAL:.+]] = stream.timepoint.await %[[SPLAT_TIMEPOINT]] => %[[SPLAT_RESULTS]]
  // CHECK: util.return %[[FINAL]]
  util.return %splat : !stream.resource<external>
}

// -----

// Tests scf.while result being used by a dispatch.
// The while produces a value consumed by a partition.

stream.async.func private @dispatch(%arg0: i64, %arg1: !stream.resource<*>) -> !stream.resource<*>

// CHECK-LABEL: @scfWhileProducingDispatchInput
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>)
util.func public @scfWhileProducingDispatchInput(%arg0: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c128 = arith.constant 128 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64

  // CHECK: %[[WHILE_RESULTS:.+]]:2 = scf.while
  // CHECK-SAME: (%[[ARG1:.+]] = %c0_i64, %[[ARG2:.+]] = %c0)
  // CHECK-SAME: (i64, index) -> (i64, index)
  %count:2 = scf.while (%iter = %c0_i64, %idx = %c0) : (i64, index) -> (i64, index) {
    %cond = arith.cmpi slt, %idx, %c10 : index
    scf.condition(%cond) %iter, %idx : i64, index
  } do {
  ^bb0(%iter: i64, %idx: index):
    %next = arith.addi %iter, %c1_i64 : i64
    %next_idx = arith.addi %idx, %c1 : index
    scf.yield %next, %next_idx : i64, index
  }

  // The dispatch should be in a partition.
  // CHECK: %[[RESULTS:.+]], %[[TIMEPOINT:.+]] = stream.async.execute
  // CHECK-SAME: with(%[[ARG0]] as %[[CAPTURE:.+]]: !stream.resource<*>{%c128})
  // CHECK-SAME: -> {{.+}}{%c128}
  // CHECK-NEXT: %[[CALL:.+]] = stream.async.call @dispatch(%[[WHILE_RESULTS]]#0, %[[CAPTURE]][%c0 to %c128 for %c128])
  // CHECK-NEXT: stream.yield %[[CALL]]
  %result = stream.async.call @dispatch(%count#0, %arg0[%c0 to %c128 for %c128]) : (i64, !stream.resource<*>{%c128}) -> %arg0{%c128}
  // CHECK: %[[FINAL:.+]] = stream.timepoint.await %[[TIMEPOINT]] => %[[RESULTS]]
  // CHECK: util.return %[[FINAL]]
  util.return %result : !stream.resource<*>
}

// -----

// Tests combination of scf.if, scf.for, and scf.while with stream operations.
// All control flow operations should remain outside partitions.

// CHECK-LABEL: @scfMixedControlFlow
util.func public @scfMixedControlFlow(%cond: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c0_i32 = arith.constant 0 : i32

  // CHECK-NOT: stream.async.execute
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c16} -> !stream.resource<external>{%c16}

  // CHECK-NOT: stream.timepoint.await
  // CHECK: scf.if
  %if_result = scf.if %cond -> i32 {
    // CHECK: scf.for
    %for_result = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter = %c0_i32) -> (i32) {
      // CHECK: stream.async.execute
      // CHECK-NEXT: stream.async.clone
      // CHECK-NEXT: stream.async.slice
      // CHECK-NEXT: stream.async.transfer
      %slice = stream.async.slice %clone[%c0 to %c4] : !stream.resource<external>{%c16} -> !stream.resource<external>{%c4}
      %transfer = stream.async.transfer %slice : !stream.resource<external>{%c4} -> !stream.resource<staging>{%c4}
      // CHECK: stream.timepoint.await
      %load = stream.async.load %transfer[%c0] : !stream.resource<staging>{%c4} -> i32
      %add = arith.addi %iter, %load : i32
      scf.yield %add : i32
    }
    scf.yield %for_result : i32
  } else {
    // CHECK: scf.while
    %while_result:2 = scf.while (%iter = %c0_i32, %count = %c0) : (i32, index) -> (i32, index) {
      %cond_while = arith.cmpi slt, %count, %c2 : index
      scf.condition(%cond_while) %iter, %count : i32, index
    } do {
    ^bb0(%iter: i32, %count: index):
      // CHECK: stream.async.execute
      // CHECK-NEXT: stream.async.clone
      // CHECK-NEXT: stream.async.slice
      // CHECK-NEXT: stream.async.transfer
      %slice = stream.async.slice %clone[%c8 to %c16] : !stream.resource<external>{%c16} -> !stream.resource<external>{%c8}
      %transfer = stream.async.transfer %slice : !stream.resource<external>{%c8} -> !stream.resource<staging>{%c8}
      // CHECK: stream.timepoint.await
      %load = stream.async.load %transfer[%c0] : !stream.resource<staging>{%c8} -> i32
      %add = arith.addi %iter, %load : i32
      %next_count = arith.addi %count, %c1 : index
      scf.yield %add, %next_count : i32, index
    }
    scf.yield %while_result#0 : i32
  }

  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.splat
  %splat = stream.async.splat %if_result : i32 -> !stream.resource<external>{%c4}
  // CHECK: stream.timepoint.await
  util.return %splat : !stream.resource<external>
}

// -----

// Tests control flow operations across multiple partitions.
// Stream operations before and after control flow should form separate partitions.

stream.async.func private @dispatch0(%arg0: !stream.resource<*>, %arg1: index) -> %arg0
stream.async.func private @dispatch1(%arg0: !stream.resource<*>, %arg1: index) -> %arg0

// CHECK-LABEL: @scfControlFlowAcrossPartitions
util.func public @scfControlFlowAcrossPartitions(%cond: i1, %arg0: !stream.resource<*>, %loop_bound: index) -> (!stream.resource<*>, i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c0_i32 = arith.constant 0 : i32

  // First partition before control flow.
  // CHECK: %[[RESULTS0:.+]], %[[TIMEPOINT0:.+]] = stream.async.execute
  // CHECK-NEXT: stream.async.call @dispatch0
  %result0 = stream.async.call @dispatch0(%arg0[%c0 to %c128 for %c128], %c128) : (!stream.resource<*>{%c128}, index) -> %arg0{%c128}

  // Control flow in the middle - loop that loads data.
  // CHECK-NOT: stream.timepoint.await
  // CHECK: scf.for
  %loop_result:2 = scf.for %i = %c0 to %loop_bound step %c1 iter_args(%iter = %result0, %sum = %c0_i32) -> (!stream.resource<*>, i32) {
    // CHECK: stream.async.execute await(%[[TIMEPOINT0]])
    // CHECK-NEXT: stream.async.transfer
    %transfer = stream.async.transfer %iter : !stream.resource<*>{%c128} -> !stream.resource<staging>{%c128}
    // CHECK: stream.timepoint.await
    %loaded = stream.async.load %transfer[%c0] : !stream.resource<staging>{%c128} -> i32
    %next_sum = arith.addi %sum, %loaded : i32
    scf.yield %iter, %next_sum : !stream.resource<*>, i32
  }

  // Second partition after control flow.
  // CHECK: stream.async.execute await(%[[TIMEPOINT0]])
  // CHECK-NEXT: stream.async.call @dispatch1
  %result1 = stream.async.call @dispatch1(%loop_result#0[%c0 to %c128 for %c128], %c128) : (!stream.resource<*>{%c128}, index) -> %loop_result#0{%c128}
  // CHECK: stream.timepoint.await
  util.return %result1, %loop_result#1 : !stream.resource<*>, i32
}

// -----

// Tests control flow with device affinities.
// Partitions should respect affinity boundaries with control flow.

util.global private @device_a : !hal.device
util.global private @device_b : !hal.device

// CHECK-LABEL: @scfControlFlowWithAffinities
util.func public @scfControlFlowWithAffinities(%cond: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c0_i32 = arith.constant 0 : i32

  // Partition on device_a.
  // CHECK-NOT: stream.async.execute
  %clone_a = stream.async.clone on(#hal.device.affinity<@device_a>) %arg0 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}

  // Control flow on host.
  // CHECK-NOT: stream.timepoint.await
  // CHECK: scf.for
  %result = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter = %c0_i32) -> (i32) {
    // Partition on device_a for clone inside loop.
    // CHECK: stream.async.execute on(#hal.device.affinity<@device_a>)
    // CHECK-NEXT: stream.async.clone
    // Partition on device_b inside loop.
    // CHECK: stream.async.execute on(#hal.device.affinity<@device_b>)
    // CHECK-NEXT: stream.async.slice
    // CHECK-NEXT: stream.async.transfer
    %slice = stream.async.slice on(#hal.device.affinity<@device_b>) %clone_a[%c0 to %c4] : !stream.resource<external>{%c8} -> !stream.resource<external>{%c4}
    %transfer = stream.async.transfer %slice : !stream.resource<external>{%c4} from(#hal.device.affinity<@device_b>) -> to(#hal.device.affinity<@device_b>) !stream.resource<staging>{%c4}
    // CHECK: stream.timepoint.await
    %load = stream.async.load %transfer[%c0] : !stream.resource<staging>{%c4} -> i32
    %add = arith.addi %iter, %load : i32
    scf.yield %add : i32
  }

  // Final partition on device_a.
  // CHECK: stream.async.execute on(#hal.device.affinity<@device_a>)
  // CHECK-NEXT: stream.async.splat
  %splat = stream.async.splat on(#hal.device.affinity<@device_a>) %result : i32 -> !stream.resource<external>{%c4}
  // CHECK: stream.timepoint.await
  util.return %splat : !stream.resource<external>
}

// -----

// Tests that operations captured in nested regions (scf.for) are not grouped
// with operations that depend on the nested region's results. This prevents
// circular dependencies where the partition would use the scf result while the
// scf captures values from the partition.

// CHECK-LABEL: @scfNestedCapture
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[ARG1:.+]]: index)
util.func public @scfNestedCapture(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c0_i64 = arith.constant 0 : i64

  // The cloned resource will be duplicated - one copy in the loop's partition,
  // one copy in the dispatch's partition. No standalone clone partition.
  %cloned = stream.async.clone %arg0 : !stream.resource<*>{%arg1} -> !stream.resource<*>{%arg1}

  // Loop gets its own materialized clone.
  // CHECK: %[[LOOP_RESULT:.+]] = scf.for
  %loop_result = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %c0_i64) -> (i64) {
    // Operations inside loop should be partitioned with cloned op materialized.
    // CHECK: stream.async.execute
    // CHECK-SAME: with(%[[ARG0]]
    // CHECK-NEXT: stream.async.clone
    // CHECK: stream.async.transfer
    %slice = stream.async.slice %cloned[%c0 to %arg1] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%arg1}
    %transfer = stream.async.transfer %slice : !stream.resource<*>{%arg1} -> !stream.resource<staging>{%arg1}
    // CHECK: stream.timepoint.await
    // CHECK: stream.async.load
    %loaded = stream.async.load %transfer[%c0] : !stream.resource<staging>{%arg1} -> i32
    %loaded_i64 = arith.extsi %loaded : i32 to i64
    %new_acc = arith.addi %acc, %loaded_i64 : i64
    scf.yield %new_acc : i64
  }

  // Dispatch doesn't use %cloned, only %loop_result, so no clone is materialized here.
  // Before the fix, partitioning would try to group %cloned with this dispatch,
  // creating a circular dependency (dispatch needs loop result, loop needs cloned).
  // After fix, %cloned is only materialized where used (in loop partition).
  // CHECK: stream.async.execute
  // CHECK-NOT: stream.async.clone
  // CHECK: stream.async.dispatch
  %dispatched = stream.async.dispatch @ex::@dispatch[%c1](%loop_result) : (i64) -> !stream.resource<*>{%arg1}

  util.return %dispatched : !stream.resource<*>
}

// -----

// Tests scf.if with divergent captures - only one branch captures from parent.
// Partitioning must be conservative and treat the containing scf.if as a hazard.

// CHECK-LABEL: @scfIfDivergentCapture
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[ARG1:.+]]: index, %[[COND:.+]]: i1)
util.func public @scfIfDivergentCapture(%arg0: !stream.resource<*>, %arg1: index, %cond: i1) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i64 = arith.constant 0 : i64

  %cloned = stream.async.clone %arg0 : !stream.resource<*>{%arg1} -> !stream.resource<*>{%arg1}

  // Only true branch captures %cloned - it will get its own materialized clone.
  // CHECK: %[[IF_RESULT:.+]] = scf.if
  %if_result = scf.if %cond -> (i64) {
    // CHECK: stream.async.execute
    // CHECK-SAME: with(%[[ARG0]]
    // CHECK-NEXT: stream.async.clone
    %slice = stream.async.slice %cloned[%c0 to %arg1] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%arg1}
    %transfer = stream.async.transfer %slice : !stream.resource<*>{%arg1} -> !stream.resource<staging>{%arg1}
    %loaded = stream.async.load %transfer[%c0] : !stream.resource<staging>{%arg1} -> i32
    %result = arith.extsi %loaded : i32 to i64
    scf.yield %result : i64
  } else {
    // False branch doesn't capture anything.
    scf.yield %c0_i64 : i64
  }

  // Dispatch doesn't use %cloned, so no clone is materialized here.
  // CHECK: stream.async.execute
  // CHECK-NOT: stream.async.clone
  // CHECK: stream.async.dispatch
  %dispatched = stream.async.dispatch @ex::@dispatch[%c1](%if_result) : (i64) -> !stream.resource<*>{%arg1}

  util.return %dispatched : !stream.resource<*>
}

// -----

// Tests nested scf.for loops where inner loop captures from grandparent scope.
// The containing operation walk should find the outer scf.for correctly.

// CHECK-LABEL: @scfNestedLoopCapture
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[ARG1:.+]]: index)
util.func public @scfNestedLoopCapture(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c0_i64 = arith.constant 0 : i64

  %cloned = stream.async.clone %arg0 : !stream.resource<*>{%arg1} -> !stream.resource<*>{%arg1}

  // Outer loop.
  // Inner loop gets its own materialized clone.
  // CHECK: %[[OUTER_RESULT:.+]] = scf.for
  %outer_result = scf.for %i = %c0 to %c5 step %c1 iter_args(%outer_acc = %c0_i64) -> (i64) {
    // Inner loop captures %cloned from grandparent block.
    // It gets its own materialized clone.
    // CHECK: scf.for
    %inner_result = scf.for %j = %c0 to %c5 step %c1 iter_args(%inner_acc = %c0_i64) -> (i64) {
      // CHECK: stream.async.execute
      // CHECK-SAME: with(%[[ARG0]]
      // CHECK-NEXT: stream.async.clone
      %slice = stream.async.slice %cloned[%c0 to %arg1] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%arg1}
      %transfer = stream.async.transfer %slice : !stream.resource<*>{%arg1} -> !stream.resource<staging>{%arg1}
      %loaded = stream.async.load %transfer[%c0] : !stream.resource<staging>{%arg1} -> i32
      %loaded_i64 = arith.extsi %loaded : i32 to i64
      %new_inner = arith.addi %inner_acc, %loaded_i64 : i64
      scf.yield %new_inner : i64
    }
    %new_outer = arith.addi %outer_acc, %inner_result : i64
    scf.yield %new_outer : i64
  }

  // Dispatch doesn't use %cloned, so no clone is materialized here.
  // CHECK: stream.async.execute
  // CHECK-NOT: stream.async.clone
  // CHECK: stream.async.dispatch
  %dispatched = stream.async.dispatch @ex::@dispatch[%c1](%outer_result) : (i64) -> !stream.resource<*>{%arg1}

  util.return %dispatched : !stream.resource<*>
}

// -----

// Tests multiple scf operations at the same level capturing the same resource.
// All should mark the resource as hazardous for grouping with their consumers.

// CHECK-LABEL: @scfMultipleCaptures
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[ARG1:.+]]: index)
util.func public @scfMultipleCaptures(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c0_i64 = arith.constant 0 : i64

  %cloned = stream.async.clone %arg0 : !stream.resource<*>{%arg1} -> !stream.resource<*>{%arg1}

  // First loop gets its own materialized clone.
  // CHECK: %[[RESULT1:.+]] = scf.for
  %result1 = scf.for %i = %c0 to %c5 step %c1 iter_args(%acc = %c0_i64) -> (i64) {
    // CHECK: stream.async.execute
    // CHECK-SAME: with(%[[ARG0]]
    // CHECK-NEXT: stream.async.clone
    %slice = stream.async.slice %cloned[%c0 to %arg1] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%arg1}
    %transfer = stream.async.transfer %slice : !stream.resource<*>{%arg1} -> !stream.resource<staging>{%arg1}
    %loaded = stream.async.load %transfer[%c0] : !stream.resource<staging>{%arg1} -> i32
    %loaded_i64 = arith.extsi %loaded : i32 to i64
    %new_acc = arith.addi %acc, %loaded_i64 : i64
    scf.yield %new_acc : i64
  }

  // Second loop also gets its own materialized clone.
  // CHECK: %[[RESULT2:.+]] = scf.for
  %result2 = scf.for %i = %c0 to %c5 step %c1 iter_args(%acc = %c0_i64) -> (i64) {
    // CHECK: stream.async.execute
    // CHECK-SAME: with(%[[ARG0]]
    // CHECK-NEXT: stream.async.clone
    %slice = stream.async.slice %cloned[%c0 to %arg1] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%arg1}
    %transfer = stream.async.transfer %slice : !stream.resource<*>{%arg1} -> !stream.resource<staging>{%arg1}
    %loaded = stream.async.load %transfer[%c0] : !stream.resource<staging>{%arg1} -> i32
    %loaded_i64 = arith.extsi %loaded : i32 to i64
    %new_acc = arith.addi %acc, %loaded_i64 : i64
    scf.yield %new_acc : i64
  }

  // Dispatch uses both loop results. Should be in separate partition from %cloned.
  // CHECK: stream.async.execute
  // CHECK-NOT: stream.async.clone
  // CHECK: stream.async.dispatch
  %sum = arith.addi %result1, %result2 : i64
  %dispatched = stream.async.dispatch @ex::@dispatch[%c1](%sum) : (i64) -> !stream.resource<*>{%arg1}

  util.return %dispatched : !stream.resource<*>
}

// -----

// Tests that nested scf operations correctly handle iter_args pattern.
// Iter_args should allow values to flow through the loop without creating hazards.

// CHECK-LABEL: @scfIterArgsPattern
util.func public @scfIterArgsPattern(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index

  // Loop with iter_args - resource flows through the loop correctly.
  // CHECK: scf.for
  // CHECK-SAME: iter_args
  %loop_result = scf.for %i = %c0 to %c5 step %c1 iter_args(%iter_resource = %arg0) -> !stream.resource<*> {
    // Each iteration partitions independently.
    // CHECK: stream.async.execute
    // CHECK: stream.async.dispatch
    %updated = stream.async.dispatch @ex::@dispatch[%c1](%iter_resource[%c0 to %arg1 for %arg1]) : (!stream.resource<*>{%arg1}) -> !stream.resource<*>{%arg1}
    // CHECK: stream.timepoint.await
    scf.yield %updated : !stream.resource<*>
  }

  util.return %loop_result : !stream.resource<*>
}

// -----

// Tests mixed iter_args (good) and captures (hazard) in same loop.
// This is a common real-world pattern where a loop updates one resource via iter_args
// while reading from another resource captured from parent scope.

// CHECK-LABEL: @scfMixedIterArgsAndCaptures
util.func public @scfMixedIterArgsAndCaptures(%arg0: !stream.resource<*>, %arg1: !stream.resource<*>, %size: index) -> (!stream.resource<*>, i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c0_i64 = arith.constant 0 : i64

  // This resource will be materialized in the loop.
  %read_only = stream.async.clone %arg1 : !stream.resource<*>{%size} -> !stream.resource<*>{%size}

  // Loop: %arg0 flows through iter_args (updated each iteration),
  //       %read_only materialized in loop (read each iteration),
  //       %accumulator tracks scalar across iterations.
  // CHECK: %[[LOOP_RESULTS:.+]]:2 = scf.for
  // CHECK-SAME: iter_args(
  %loop_resource, %loop_sum = scf.for %i = %c0 to %c5 step %c1
      iter_args(%iter_resource = %arg0, %iter_sum = %c0_i64) -> (!stream.resource<*>, i64) {

    // Inside loop: both iter_arg resource and materialized clone are used.
    // They must be in the same execute region since the dispatch needs the
    // updated iter_arg, and the transfer needs the materialized clone.
    // CHECK: stream.async.execute
    // CHECK: stream.async.clone
    // CHECK: stream.async.dispatch
    // CHECK: stream.async.transfer
    %updated = stream.async.dispatch @ex::@dispatch[%c1](%iter_resource[%c0 to %size for %size])
        : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}

    %slice = stream.async.slice %read_only[%c0 to %size] : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
    %transfer = stream.async.transfer %slice : !stream.resource<*>{%size} -> !stream.resource<staging>{%size}
    %loaded = stream.async.load %transfer[%c0] : !stream.resource<staging>{%size} -> i32
    %loaded_i64 = arith.extsi %loaded : i32 to i64
    %new_sum = arith.addi %iter_sum, %loaded_i64 : i64

    scf.yield %updated, %new_sum : !stream.resource<*>, i64
  }

  // The key test: the final dispatch uses loop results but NOT the captured
  // %read_only. This proves that %read_only is correctly tracked as a hazard
  // and not incorrectly merged with operations that only use the loop's outputs.
  // CHECK: stream.async.execute
  // CHECK-NOT: stream.async.clone
  // CHECK: stream.async.dispatch
  %final = stream.async.dispatch @ex::@dispatch[%c1](%loop_sum, %loop_resource[%c0 to %size for %size])
      : (i64, !stream.resource<*>{%size}) -> !stream.resource<*>{%size}

  util.return %final, %loop_sum : !stream.resource<*>, i64
}

// -----

// Tests splat with optimization_barrier in same block + use inside scf.if.
// This reproduces the bug from if.mlir: splat with barrier use AND use in nested
// region was not being placed in any partition, causing verification failure.
// The barrier forces the splat into a partition before the scf.if.

// CHECK-LABEL: @splatWithBarrierAndScfIfUse
util.func public @splatWithBarrierAndScfIfUse(%cond: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c10_i32 = arith.constant 10 : i32
  %c4 = arith.constant 4 : index

  // CHECK: %[[RESULTS:.+]], %[[TIMEPOINT:.+]] = stream.async.execute
  // CHECK-SAME: with() -> !stream.resource<external>{%c4}
  // CHECK-NEXT: %[[SPLAT:.+]] = stream.async.splat %c10_i32
  // CHECK-NEXT: stream.yield %[[SPLAT]]
  %splat = stream.async.splat %c10_i32 : i32 -> !stream.resource<external>{%c4}

  // Same-block non-streamable user forces await before scf.if.
  // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[TIMEPOINT]]
  %barrier = util.optimization_barrier %splat : !stream.resource<external>

  // CHECK: scf.if
  %result = scf.if %cond -> !stream.resource<external> {
    // Nested region can use both the awaited value and reference original.
    // CHECK: stream.async.execute
    // CHECK: stream.async.dispatch
    %dispatch = stream.async.dispatch @dispatch::@entry(%barrier[%c0 to %c4 for %c4], %splat[%c0 to %c4 for %c4]) : (!stream.resource<external>{%c4}, !stream.resource<external>{%c4}) -> !stream.resource<external>{%c4}
    scf.yield %dispatch : !stream.resource<external>
  } else {
    scf.yield %barrier : !stream.resource<external>
  }

  util.return %result : !stream.resource<external>
}

stream.executable private @dispatch {
  stream.executable.export public @entry
  builtin.module {
    func.func @entry(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) {
      return
    }
  }
}

// -----

// Tests clone with optimization_barrier in same block + use inside scf.for.
// Similar to test above but with clone + scf.for instead of splat + scf.if.

// CHECK-LABEL: @cloneWithBarrierAndScfForUse
util.func public @cloneWithBarrierAndScfForUse(%arg0: !stream.resource<external>, %ub: index) -> i64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c0_i64 = arith.constant 0 : i64

  // CHECK: %[[RESULTS:.+]], %[[TIMEPOINT:.+]] = stream.async.execute
  // CHECK-SAME: with(%{{.+}} as %[[ARG:.+]]: !stream.resource<external>{%c8})
  // CHECK-SAME: -> !stream.resource<external>{%c8}
  // CHECK-NEXT: %[[CLONE:.+]] = stream.async.clone %[[ARG]]
  // CHECK-NEXT: stream.yield %[[CLONE]]
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}

  // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[TIMEPOINT]]
  %barrier = util.optimization_barrier %clone : !stream.resource<external>

  // CHECK: scf.for
  %result = scf.for %i = %c0 to %ub step %c1 iter_args(%iter = %c0_i64) -> (i64) {
    // CHECK: stream.async.execute
    // CHECK: stream.async.slice
    // CHECK: stream.async.transfer
    %results_0, %result_timepoint_1 = stream.async.execute with(%barrier as %arg1: !stream.resource<external>{%c8}) -> !stream.resource<staging>{%c4} {
      %3 = stream.async.slice %arg1[%c0 to %c4] : !stream.resource<external>{%c8} -> !stream.resource<external>{%c4}
      %4 = stream.async.transfer %3 : !stream.resource<external>{%c4} -> !stream.resource<staging>{%c4}
      stream.yield %4 : !stream.resource<staging>{%c4}
    } => !stream.timepoint
    %1 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<staging>{%c4}
    %2 = stream.async.load %1[%c0] : !stream.resource<staging>{%c4} -> i32
    %ext = arith.extsi %2 : i32 to i64
    %next = arith.addi %iter, %ext : i64
    scf.yield %next : i64
  }

  util.return %result : i64
}

// -----

// Tests splat with multiple optimization_barrier uses in same block.
// Ensures that multiple same-block non-streamable users work correctly.

// CHECK-LABEL: @splatWithMultipleBarriers
util.func public @splatWithMultipleBarriers() -> (!stream.resource<external>, !stream.resource<external>) {
  %c10_i32 = arith.constant 10 : i32
  %c4 = arith.constant 4 : index

  // CHECK: %[[RESULTS:.+]], %[[TIMEPOINT:.+]] = stream.async.execute
  // CHECK-NEXT: %[[SPLAT:.+]] = stream.async.splat
  // CHECK-NEXT: stream.yield %[[SPLAT]]
  %splat = stream.async.splat %c10_i32 : i32 -> !stream.resource<external>{%c4}

  // Both barriers use the same awaited value.
  // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[TIMEPOINT]]
  // CHECK: util.optimization_barrier %[[AWAITED]]
  %barrier1 = util.optimization_barrier %splat : !stream.resource<external>
  // CHECK: util.optimization_barrier %[[AWAITED]]
  %barrier2 = util.optimization_barrier %splat : !stream.resource<external>

  util.return %barrier1, %barrier2 : !stream.resource<external>, !stream.resource<external>
}

// -----

// Tests clone with optimization_barrier + use in deeply nested regions.
// Ensures barrier causes partitioning even with deep nesting (scf.for > scf.if).

// CHECK-LABEL: @clonableBarrierDeeplyNested
util.func public @clonableBarrierDeeplyNested(%arg0: !stream.resource<external>, %cond: i1, %ub: index) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // CHECK: %[[RESULTS:.+]], %[[TIMEPOINT:.+]] = stream.async.execute
  // CHECK-SAME: with(%{{.+}} as %[[ARG:.+]]: !stream.resource<external>{%c8})
  // CHECK-SAME: -> !stream.resource<external>{%c8}
  // CHECK-NEXT: %[[CLONE:.+]] = stream.async.clone %[[ARG]]
  // CHECK-NEXT: stream.yield %[[CLONE]]
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}

  // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[TIMEPOINT]]
  %barrier = util.optimization_barrier %clone : !stream.resource<external>

  // CHECK: scf.for
  %result = scf.for %i = %c0 to %ub step %c1 iter_args(%iter = %arg0) -> !stream.resource<external> {
    // CHECK: scf.if
    %next = scf.if %cond -> !stream.resource<external> {
      // Deeply nested use of barrier.
      // CHECK: stream.async.execute
      // CHECK: stream.async.slice
      %results_0, %result_timepoint_1 = stream.async.execute with(%barrier as %arg1: !stream.resource<external>{%c8}) -> !stream.resource<external>{%c4} {
        %2 = stream.async.slice %arg1[%c0 to %c4] : !stream.resource<external>{%c8} -> !stream.resource<external>{%c4}
        stream.yield %2 : !stream.resource<external>{%c4}
      } => !stream.timepoint
      %1 = stream.timepoint.await %result_timepoint_1 => %results_0 : !stream.resource<external>{%c4}
      scf.yield %1 : !stream.resource<external>
    } else {
      scf.yield %iter : !stream.resource<external>
    }
    scf.yield %next : !stream.resource<external>
  }

  util.return %result : !stream.resource<external>
}

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

  // CHECK: %[[CLONE:.+]], %[[CLONE_TIMEPOINT:.+]] = stream.async.execute
  // CHECK-SAME: with(%{{.+}} as %[[CLONE_CAPTURE:.+]]: !stream.resource<external>{%c48})
  // CHECK-SAME: -> !stream.resource<external>{%c48}
  // CHECK-NEXT: %[[CLONE_OP:.+]] = stream.async.clone %[[CLONE_CAPTURE]]
  // CHECK-NEXT: stream.yield %[[CLONE_OP]]
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c48} -> !stream.resource<external>{%c48}

  // CHECK-NOT: stream.timepoint.await
  // CHECK: scf.for
  %result = scf.for %i = %c0 to %c11 step %c1 iter_args(%iter = %c0_i64) -> (i64) {
    %offset = arith.muli %i, %c4 : index
    %end = arith.addi %offset, %c4 : index
    // Loop uses the cloned resource - this should NOT be partitioned into the clone partition.
    // CHECK: stream.async.execute await(%[[CLONE_TIMEPOINT]])
    // CHECK-NEXT: stream.async.slice
    // CHECK-NEXT: stream.async.transfer
    %slice = stream.async.slice %clone[%offset to %end] : !stream.resource<external>{%c48} -> !stream.resource<external>{%c4}
    %transfer = stream.async.transfer %slice : !stream.resource<external>{%c4} -> !stream.resource<staging>{%c4}
    // CHECK: stream.timepoint.await
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

// Tests scf.for loop using multiple values from partition.
// The loop body uses multiple partition outputs as inputs.

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

  // CHECK: %[[CLONE:.+]], %[[CLONE_TIMEPOINT:.+]] = stream.async.execute
  // CHECK-NEXT: stream.async.clone
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c16} -> !stream.resource<external>{%c16}

  // CHECK-NOT: stream.timepoint.await
  // CHECK: scf.for
  %result = scf.for %i = %lb to %ub step %c1 iter_args(%iter = %c0_i32) -> (i32) {
    // CHECK: stream.async.execute await(%[[CLONE_TIMEPOINT]])
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
// scf.if tests
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
    // CHECK: stream.async.execute
    // CHECK-NEXT: stream.async.slice
    %slice = stream.async.slice %arg0[%c0 to %c8] : !stream.resource<external>{%c16} -> !stream.resource<external>{%c8}
    // CHECK: stream.timepoint.await
    scf.yield %slice : !stream.resource<external>
  } else {
    // CHECK: stream.async.execute
    // CHECK-NEXT: stream.async.slice
    %slice = stream.async.slice %arg0[%c8 to %c16] : !stream.resource<external>{%c16} -> !stream.resource<external>{%c8}
    // CHECK: stream.timepoint.await
    scf.yield %slice : !stream.resource<external>
  }

  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.clone
  %transfer = stream.async.transfer %result : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}
  // CHECK: stream.timepoint.await
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
    // CHECK: stream.async.execute
    // CHECK-NEXT: stream.async.slice
    %slice = stream.async.slice %arg0[%c0 to %c4] : !stream.resource<external>{%c8} -> !stream.resource<external>{%c4}
    // CHECK: stream.timepoint.await
    scf.yield %slice : !stream.resource<external>
  } else {
    scf.yield %arg0 : !stream.resource<external>
  }

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

  // CHECK: %[[CLONE:.+]], %[[CLONE_TIMEPOINT:.+]] = stream.async.execute
  // CHECK-SAME: with(%{{.+}} as %[[CLONE_CAPTURE:.+]]: !stream.resource<external>{%c8})
  // CHECK-SAME: -> !stream.resource<external>{%c8}
  // CHECK-NEXT: %[[CLONE_OP:.+]] = stream.async.clone %[[CLONE_CAPTURE]]
  // CHECK-NEXT: stream.yield %[[CLONE_OP]]
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}

  // The else branch uses clone directly (not in a stream op), so await must happen before scf.if.
  // CHECK: %[[CLONE_READY:.+]] = stream.timepoint.await %[[CLONE_TIMEPOINT]] => %[[CLONE]]
  // CHECK: scf.if
  %result = scf.if %cond -> !stream.resource<external> {
    // Then branch uses cloned resource in stream op - can reference original clone.
    // CHECK: stream.async.execute await(%[[CLONE_TIMEPOINT]])
    // CHECK-NEXT: stream.async.slice
    %slice = stream.async.slice %clone[%c0 to %c4] : !stream.resource<external>{%c8} -> !stream.resource<external>{%c4}
    // CHECK: stream.timepoint.await
    scf.yield %slice : !stream.resource<external>
  } else {
    scf.yield %clone : !stream.resource<external>
  }

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
  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.call @dispatch
  %result = stream.async.call @dispatch(%arg0[%c0 to %size for %size], %size) : (!stream.resource<*>{%size}, index) -> %arg0{%size}
  // CHECK: stream.timepoint.await
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
      // CHECK: stream.async.execute
      // CHECK-NEXT: stream.async.slice
      // CHECK-NEXT: stream.async.transfer
      %slice = stream.async.slice %arg0[%c0 to %c4] : !stream.resource<external>{%c16} -> !stream.resource<external>{%c4}
      %transfer = stream.async.transfer %slice : !stream.resource<external>{%c4} -> !stream.resource<staging>{%c4}
      // CHECK: stream.timepoint.await
      %load = stream.async.load %transfer[%c0] : !stream.resource<staging>{%c4} -> i32
      scf.yield %load : i32
    } else {
      // CHECK: stream.async.execute
      // CHECK-NEXT: stream.async.slice
      // CHECK-NEXT: stream.async.transfer
      %slice = stream.async.slice %arg0[%c8 to %c16] : !stream.resource<external>{%c16} -> !stream.resource<external>{%c8}
      %transfer = stream.async.transfer %slice : !stream.resource<external>{%c8} -> !stream.resource<staging>{%c8}
      // CHECK: stream.timepoint.await
      %load = stream.async.load %transfer[%c0] : !stream.resource<staging>{%c8} -> i32
      scf.yield %load : i32
    }

    %add = arith.addi %iter, %value : i32
    scf.yield %add : i32
  }

  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.splat
  %splat = stream.async.splat %result : i32 -> !stream.resource<external>{%c4}
  util.return %splat : !stream.resource<external>
}

// -----
// scf.while tests
// -----

// Tests basic scf.while loop with stream operations in before/after regions.
// Both regions should be partitioned independently.

// CHECK-LABEL: @scfWhileBasic
util.func public @scfWhileBasic(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c10 = arith.constant 10 : index
  %c0_i32 = arith.constant 0 : i32
  %c100_i32 = arith.constant 100 : i32

  // CHECK: scf.while
  %result:2 = scf.while (%iter = %c0_i32, %count = %c0) : (i32, index) -> (i32, index) {
    %cond = arith.cmpi slt, %count, %c10 : index
    // CHECK: scf.condition
    scf.condition(%cond) %iter, %count : i32, index
  } do {
  ^bb0(%iter: i32, %count: index):
    // CHECK: stream.async.execute
    // CHECK-NEXT: stream.async.slice
    // CHECK-NEXT: stream.async.transfer
    %slice = stream.async.slice %arg0[%c0 to %c4] : !stream.resource<external>{%c8} -> !stream.resource<external>{%c4}
    %transfer = stream.async.transfer %slice : !stream.resource<external>{%c4} -> !stream.resource<staging>{%c4}
    // CHECK: stream.timepoint.await
    %load = stream.async.load %transfer[%c0] : !stream.resource<staging>{%c4} -> i32
    %add = arith.addi %iter, %load : i32
    %next_count = arith.addi %count, %c1 : index
    scf.yield %add, %next_count : i32, index
  }

  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.splat
  %splat = stream.async.splat %result#0 : i32 -> !stream.resource<external>{%c4}
  util.return %splat : !stream.resource<external>
}

// -----

// Tests scf.while loop using cloned resource from partition.
// The while depends on a partition output.

// CHECK-LABEL: @scfWhileWithDependency
util.func public @scfWhileWithDependency(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c5 = arith.constant 5 : index
  %c0_i32 = arith.constant 0 : i32

  // CHECK: %[[CLONE:.+]], %[[CLONE_TIMEPOINT:.+]] = stream.async.execute
  // CHECK-NEXT: stream.async.clone
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}

  // CHECK-NOT: stream.timepoint.await
  // CHECK: scf.while
  %result:2 = scf.while (%iter = %c0_i32, %count = %c0) : (i32, index) -> (i32, index) {
    %cond = arith.cmpi slt, %count, %c5 : index
    scf.condition(%cond) %iter, %count : i32, index
  } do {
  ^bb0(%iter: i32, %count: index):
    // While uses the cloned resource.
    // CHECK: stream.async.execute await(%[[CLONE_TIMEPOINT]])
    // CHECK-NEXT: stream.async.slice
    // CHECK-NEXT: stream.async.transfer
    %slice = stream.async.slice %clone[%c0 to %c4] : !stream.resource<external>{%c8} -> !stream.resource<external>{%c4}
    %transfer = stream.async.transfer %slice : !stream.resource<external>{%c4} -> !stream.resource<staging>{%c4}
    // CHECK: stream.timepoint.await
    %load = stream.async.load %transfer[%c0] : !stream.resource<staging>{%c4} -> i32
    %add = arith.addi %iter, %load : i32
    %next_count = arith.addi %count, %c1 : index
    scf.yield %add, %next_count : i32, index
  }

  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.splat
  %splat = stream.async.splat %result#0 : i32 -> !stream.resource<external>{%c4}
  util.return %splat : !stream.resource<external>
}

// -----

// Tests scf.while result being used by a dispatch.
// The while produces a value consumed by a partition.

stream.async.func private @dispatch(%arg0: i64, %arg1: !stream.resource<*>) -> !stream.resource<*>

// CHECK-LABEL: @scfWhileProducingDispatchInput
util.func public @scfWhileProducingDispatchInput(%arg0: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c128 = arith.constant 128 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64

  // CHECK: scf.while
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
  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.call @dispatch
  %result = stream.async.call @dispatch(%count#0, %arg0[%c0 to %c128 for %c128]) : (i64, !stream.resource<*>{%c128}) -> %arg0{%c128}
  // CHECK: stream.timepoint.await
  util.return %result : !stream.resource<*>
}

// -----
// Mixed control flow tests
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

  // CHECK: %[[CLONE:.+]], %[[CLONE_TIMEPOINT:.+]] = stream.async.execute
  // CHECK-NEXT: stream.async.clone
  %clone = stream.async.clone %arg0 : !stream.resource<external>{%c16} -> !stream.resource<external>{%c16}

  // CHECK-NOT: stream.timepoint.await
  // CHECK: scf.if
  %if_result = scf.if %cond -> i32 {
    // CHECK: scf.for
    %for_result = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter = %c0_i32) -> (i32) {
      // CHECK: stream.async.execute await(%[[CLONE_TIMEPOINT]])
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
      // CHECK: stream.async.execute await(%[[CLONE_TIMEPOINT]])
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
  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.call @dispatch0
  %result0 = stream.async.call @dispatch0(%arg0[%c0 to %c128 for %c128], %c128) : (!stream.resource<*>{%c128}, index) -> %arg0{%c128}

  // Control flow in the middle - loop that loads data.
  // CHECK-NOT: stream.timepoint.await
  // CHECK: scf.for
  %loop_result:2 = scf.for %i = %c0 to %loop_bound step %c1 iter_args(%iter = %result0, %sum = %c0_i32) -> (!stream.resource<*>, i32) {
    // CHECK: stream.async.execute
    // CHECK-NEXT: stream.async.transfer
    %transfer = stream.async.transfer %iter : !stream.resource<*>{%c128} -> !stream.resource<staging>{%c128}
    // CHECK: stream.timepoint.await
    %loaded = stream.async.load %transfer[%c0] : !stream.resource<staging>{%c128} -> i32
    %next_sum = arith.addi %sum, %loaded : i32
    scf.yield %iter, %next_sum : !stream.resource<*>, i32
  }

  // Second partition after control flow.
  // CHECK: stream.async.execute
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
  // CHECK: stream.async.execute on(#hal.device.affinity<@device_a>)
  // CHECK-NEXT: stream.async.clone
  %clone_a = stream.async.clone on(#hal.device.affinity<@device_a>) %arg0 : !stream.resource<external>{%c8} -> !stream.resource<external>{%c8}

  // Control flow on host.
  // CHECK-NOT: stream.timepoint.await
  // CHECK: scf.for
  %result = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter = %c0_i32) -> (i32) {
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
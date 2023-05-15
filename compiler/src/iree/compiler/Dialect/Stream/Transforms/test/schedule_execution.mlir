// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-stream-schedule-execution))" %s | FileCheck %s

// Tests basic partitioning of multiple ops.

// CHECK-LABEL: @partitioning
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<external>, %[[ARG1:.+]]: !stream.resource<external>)
func.func @partitioning(%arg0: !stream.resource<external>, %arg1: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %c80 = arith.constant 80 : index
  %c1280 = arith.constant 1280 : index
  %c255_i32 = arith.constant 255 : i32
  // CHECK: %[[RESULT:.+]], %[[TIMEPOINT:.+]] = stream.async.execute
  // CHECK-SAME: with(%[[ARG1]] as %[[ARG1_CAPTURE:.+]]: !stream.resource<external>{%c80},
  // CHECK-SAME:      %[[ARG0]] as %[[ARG0_CAPTURE:.+]]: !stream.resource<external>{%c20})
  // CHECK-SAME: -> !stream.resource<external>{%c20} {
  // CHECK-NEXT: %[[SPLAT0:.+]] = stream.async.splat
  %2 = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%c1280}
  // CHECK-NEXT: %[[DISPATCH0:.+]] = stream.async.dispatch @ex::@dispatch_0[%c1, %c1, %c1](%[[SPLAT0]][{{.+}}], %[[ARG1_CAPTURE]][{{.+}}]) : (!stream.resource<transient>{%c1280}, !stream.resource<external>{%c80}) -> %[[SPLAT0]]{%c1280}
  %3 = stream.async.dispatch @ex::@dispatch_0[%c1, %c1, %c1](%2[%c0 to %c1280 for %c1280], %arg1[%c0 to %c80 for %c80]) : (!stream.resource<transient>{%c1280}, !stream.resource<external>{%c80}) -> %2{%c1280}
  // CHECK-NEXT: %[[SPLAT1:.+]] = stream.async.splat
  %4 = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%c20}
  // CHECK-NEXT: %[[DISPATCH1:.+]] = stream.async.dispatch @ex::@dispatch_1[%c1, %c1, %c1](%[[ARG0_CAPTURE]][{{.+}}], %[[SPLAT1]][{{.+}}]) : (!stream.resource<external>{%c20}, !stream.resource<transient>{%c20}) -> %[[SPLAT1]]{%c20}
  %5 = stream.async.dispatch @ex::@dispatch_1[%c1, %c1, %c1](%arg0[%c0 to %c20 for %c20], %4[%c0 to %c20 for %c20]) : (!stream.resource<external>{%c20}, !stream.resource<transient>{%c20}) -> %4{%c20}
  // CHECK-NEXT: %[[DISPATCH2:.+]] = stream.async.dispatch @ex::@dispatch_2[%c1, %c1, %c1](%[[DISPATCH0]][{{.+}}], %[[DISPATCH1]][{{.+}}]) : (!stream.resource<transient>{%c1280}, !stream.resource<transient>{%c20}) -> !stream.resource<external>{%c20}
  %6 = stream.async.dispatch @ex::@dispatch_2[%c1, %c1, %c1](%3[%c0 to %c1280 for %c1280], %5[%c0 to %c20 for %c20]) : (!stream.resource<transient>{%c1280}, !stream.resource<transient>{%c20}) -> !stream.resource<external>{%c20}
  // CHECK-NEXT: stream.yield %[[DISPATCH2]] : !stream.resource<external>{%c20}
  // CHECK-NEXT: } => !stream.timepoint
  // CHECK-NEXT: %[[READY:.+]] = stream.timepoint.await %[[TIMEPOINT]] => %[[RESULT]] : !stream.resource<external>{%c20}
  // CHECK-NEXT: return %[[READY]]
  return %6 : !stream.resource<external>
}

// -----

// Tests basic partitioning of sequential dispatches with differing affinities.
// Dispatches with the same affinities should be placed into the same execution
// regions.

// CHECK-LABEL: @partitioningWithAffinities
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<external>)
func.func @partitioningWithAffinities(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %c1280 = arith.constant 1280 : index
  %c255_i32 = arith.constant 255 : i32

  // CHECK: %[[TRANSIENTS:.+]]:2, %[[TIMEPOINT0:.+]] = stream.async.execute
  // CHECK-SAME: on(#hal.affinity.queue<[0]>)
  // CHECK-SAME: with(%[[ARG0]] as %[[ARG0_CAPTURE:.+]]: !stream.resource<external>{%c20})
  // CHECK-SAME: -> (!stream.resource<transient>{%c1280}, !stream.resource<transient>{%c20}) {
  // CHECK-NEXT: %[[SPLAT:.+]] = stream.async.splat
  %splat = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%c1280}
  // CHECK-NEXT: %[[DISPATCH0:.+]] = stream.async.dispatch @ex::@dispatch_0[%c1](%[[ARG0_CAPTURE]][{{.+}}], %[[SPLAT]][{{.+}}])
  %dispatch0 = stream.async.dispatch on(#hal.affinity.queue<[0]>) @ex::@dispatch_0[%c1](%arg0[%c0 to %c20 for %c20], %splat[%c0 to %c20 for %c20]) : (!stream.resource<external>{%c20}, !stream.resource<transient>{%c20}) -> !stream.resource<transient>{%c1280}
  // CHECK-NEXT: %[[DISPATCH1:.+]] = stream.async.dispatch @ex::@dispatch_1[%c1](%[[ARG0_CAPTURE]][{{.+}}], %[[SPLAT]][{{.+}}])
  %dispatch1 = stream.async.dispatch on(#hal.affinity.queue<[0]>) @ex::@dispatch_1[%c1](%arg0[%c0 to %c20 for %c20], %splat[%c0 to %c20 for %c20]) : (!stream.resource<external>{%c20}, !stream.resource<transient>{%c20}) -> !stream.resource<transient>{%c20}
  // CHECK-NEXT: stream.yield %[[DISPATCH0]], %[[DISPATCH1]]
  // CHECK-NEXT: } => !stream.timepoint

  // CHECK: %[[RESULT:.+]], %[[TIMEPOINT1:.+]] = stream.async.execute
  // CHECK-SAME: on(#hal.affinity.queue<[1]>)
  // CHECK-SAME: await(%[[TIMEPOINT0]])
  // CHECK-SAME: with(%[[TRANSIENTS]]#0 as %[[TRANSIENT0_CAPTURE:.+]]: !stream.resource<transient>{%c1280},
  // CHECK-SAME:      %[[TRANSIENTS]]#1 as %[[TRANSIENT1_CAPTURE:.+]]: !stream.resource<transient>{%c20})
  // CHECK-SAME: -> !stream.resource<external>{%c20}
  // CHECK-NEXT: %[[DISPATCH2:.+]] = stream.async.dispatch @ex::@dispatch_2[%c1](%[[TRANSIENT0_CAPTURE]][{{.+}}], %[[TRANSIENT1_CAPTURE]][{{.+}}])
  %dispatch2 = stream.async.dispatch on(#hal.affinity.queue<[1]>) @ex::@dispatch_2[%c1](%dispatch0[%c0 to %c1280 for %c1280], %dispatch1[%c0 to %c20 for %c20]) : (!stream.resource<transient>{%c1280}, !stream.resource<transient>{%c20}) -> !stream.resource<external>{%c20}
  // CHECK-NEXT: stream.yield %[[DISPATCH2]]
  // CHECK-NEXT: } => !stream.timepoint

  // CHECK-NEXT: %[[READY:.+]] = stream.timepoint.await
  // CHECK-SAME:   on(#hal.affinity.queue<[1]>)
  // CHECK-SAME:   %[[TIMEPOINT1]] => %[[RESULT]] : !stream.resource<external>{%c20}
  // CHECK-NEXT: return %[[READY]]
  return %dispatch2 : !stream.resource<external>
}

// -----

// Tests partitioning of dispatches with differing affinities and no data
// dependencies. Unrelated dispatches with differing affinities should end up
// in concurrently executable regions.

// CHECK-LABEL: @partitioningWithConcurrentAffinities
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<external>)
func.func @partitioningWithConcurrentAffinities(%arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %c1280 = arith.constant 1280 : index
  %c255_i32 = arith.constant 255 : i32

  // CHECK: %[[TRANSIENT0:.+]], %[[TIMEPOINT0:.+]] = stream.async.execute
  // CHECK-SAME: on(#hal.affinity.queue<[0]>)
  // CHECK-SAME: with(%[[ARG0]] as %[[ARG0_CAPTURE0:.+]]: !stream.resource<external>{%c20})
  // CHECK-SAME: !stream.resource<transient>{%c1280}
  // CHECK-NEXT: %[[SPLAT0:.+]] = stream.async.splat
  %splat = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%c1280}
  // CHECK-NEXT: %[[DISPATCH0:.+]] = stream.async.dispatch @ex::@dispatch_0[%c1](%[[ARG0_CAPTURE0]][{{.+}}], %[[SPLAT0]][{{.+}}])
  %dispatch0 = stream.async.dispatch on(#hal.affinity.queue<[0]>) @ex::@dispatch_0[%c1](%arg0[%c0 to %c20 for %c20], %splat[%c0 to %c20 for %c20]) : (!stream.resource<external>{%c20}, !stream.resource<transient>{%c20}) -> !stream.resource<transient>{%c1280}
  // CHECK-NEXT: stream.yield %[[DISPATCH0]]
  // CHECK-NEXT: } => !stream.timepoint

  // CHECK: %[[TRANSIENT1:.+]], %[[TIMEPOINT1:.+]] = stream.async.execute
  // CHECK-SAME: on(#hal.affinity.queue<[1]>)
  // CHECK-SAME: with(%[[ARG0]] as %[[ARG0_CAPTURE1:.+]]: !stream.resource<external>{%c20})
  // CHECK-SAME: -> !stream.resource<transient>{%c20} {
  // CHECK-NEXT: %[[SPLAT1:.+]] = stream.async.splat
  // CHECK-NEXT: %[[DISPATCH1:.+]] = stream.async.dispatch @ex::@dispatch_1[%c1](%[[ARG0_CAPTURE1]][{{.+}}], %[[SPLAT1]][{{.+}}])
  %dispatch1 = stream.async.dispatch on(#hal.affinity.queue<[1]>) @ex::@dispatch_1[%c1](%arg0[%c0 to %c20 for %c20], %splat[%c0 to %c20 for %c20]) : (!stream.resource<external>{%c20}, !stream.resource<transient>{%c20}) -> !stream.resource<transient>{%c20}
  // CHECK-NEXT: stream.yield %[[DISPATCH1]]
  // CHECK-NEXT: } => !stream.timepoint

  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[TIMEPOINT0]], %[[TIMEPOINT1]])

  // CHECK: %[[RESULT:.+]], %[[TIMEPOINT2:.+]] = stream.async.execute
  // CHECK-SAME: await(%[[JOIN]])
  // CHECK-SAME: with(%[[TRANSIENT0]] as %[[TRANSIENT0_CAPTURE:.+]]: !stream.resource<transient>{%c1280},
  // CHECK-SAME:      %[[TRANSIENT1]] as %[[TRANSIENT1_CAPTURE:.+]]: !stream.resource<transient>{%c20})
  // CHECK-NEXT: %[[DISPATCH2:.+]] = stream.async.dispatch @ex::@dispatch_2[%c1](%[[TRANSIENT0_CAPTURE]][{{.+}}], %[[TRANSIENT1_CAPTURE]][{{.+}}])
  %dispatch2 = stream.async.dispatch on(#hal.affinity.queue<[2]>) @ex::@dispatch_2[%c1](%dispatch0[%c0 to %c1280 for %c1280], %dispatch1[%c0 to %c20 for %c20]) : (!stream.resource<transient>{%c1280}, !stream.resource<transient>{%c20}) -> !stream.resource<external>{%c20}
  // CHECK-NEXT: stream.yield %[[DISPATCH2]]
  // CHECK-NEXT: } => !stream.timepoint

  // CHECK-NEXT: %[[READY:.+]] = stream.timepoint.await
  // CHECK-SAME:   on(#hal.affinity.queue<[2]>)
  // CHECK-SAME:   %[[TIMEPOINT2]] => %[[RESULT]] : !stream.resource<external>{%c20}
  // CHECK-NEXT: return %[[READY]]
  return %dispatch2 : !stream.resource<external>
}

// -----

// Tests that ops in multiple blocks are partitioned independently and that
// timepoints are chained between the partitions. Note that the dispatches
// happen in-place on the splat and we expect the execution regions to be tied.

// CHECK-LABEL: @partitionWithinBlocks
func.func @partitionWithinBlocks(%cond: i1) -> !stream.resource<transient> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1280 = arith.constant 1280 : index
  %c255_i32 = arith.constant 255 : i32
  // CHECK: %[[SPLAT:.+]], %[[SPLAT_TIMEPOINT:.+]] = stream.async.execute
  // CHECK: stream.async.splat
  %splat = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%c1280}
  // CHECK: cf.cond_br
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  // CHECK: %[[BB1_RESULT:.+]], %[[BB1_TIMEPOINT:.+]] = stream.async.execute await(%[[SPLAT_TIMEPOINT]]) =>
  // CHECK-SAME: with(%[[SPLAT]] as %[[BB1_SPLAT:.+]]: !stream.resource<transient>{%c1280})
  // CHECK-SAME: -> %[[SPLAT]]{%c1280}
  // CHECK: stream.async.dispatch @ex::@dispatch_0[%c1, %c1, %c1](%[[BB1_SPLAT]][{{.+}}]) : (!stream.resource<transient>{%c1280}) -> %[[BB1_SPLAT]]{%c1280}
  %3 = stream.async.dispatch @ex::@dispatch_0[%c1, %c1, %c1](%splat[%c0 to %c1280 for %c1280]) : (!stream.resource<transient>{%c1280}) -> %splat{%c1280}
  // CHECK: %[[BB1_READY:.+]] = stream.timepoint.await %[[BB1_TIMEPOINT]] => %[[BB1_RESULT]]
  // CHECK: return %[[BB1_READY]]
  return %3 : !stream.resource<transient>
^bb2:
  // CHECK: %[[BB2_RESULT:.+]], %[[BB2_TIMEPOINT:.+]] = stream.async.execute await(%[[SPLAT_TIMEPOINT]]) =>
  // CHECK-SAME: with(%[[SPLAT]] as %[[BB2_SPLAT:.+]]: !stream.resource<transient>{%c1280})
  // CHECK-SAME: -> %[[SPLAT]]{%c1280}
  // CHECK: stream.async.dispatch @ex::@dispatch_1[%c1, %c1, %c1](%[[BB2_SPLAT]][{{.+}}]) : (!stream.resource<transient>{%c1280}) -> %[[BB2_SPLAT]]{%c1280}
  %4 = stream.async.dispatch @ex::@dispatch_1[%c1, %c1, %c1](%splat[%c0 to %c1280 for %c1280]) : (!stream.resource<transient>{%c1280}) -> %splat{%c1280}
  // CHECK: %[[BB2_READY:.+]] = stream.timepoint.await %[[BB2_TIMEPOINT]] => %[[BB2_RESULT]]
  // CHECK: return %[[BB2_READY]]
  return %4 : !stream.resource<transient>
}

// -----

// Tests a complex device->host->device sequence gets turned into the proper
// execute->await->execute. These data-dependent operations can happen in a
// single block and break the assumption that one block == one partition.

// CHECK-LABEL: @deviceHostDevice
func.func @deviceHostDevice() -> !stream.resource<transient> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c123_i8 = arith.constant 123 : i8
  // CHECK: %[[RESULT_D2H:.+]], %[[TIMEPOINT_D2H:.+]] = stream.async.execute with()
  // CHECK-SAME: -> !stream.resource<staging>{%c1}
  // CHECK-NEXT: %[[SPLAT:.+]] = stream.async.splat %c123_i8
  %0 = stream.async.splat %c123_i8 : i8 -> !stream.resource<transient>{%c1}
  // CHECK-NEXT: %[[TRANSFER_D2H:.+]] = stream.async.transfer %[[SPLAT]]
  %1 = stream.async.transfer %0 : !stream.resource<transient>{%c1} -> !stream.resource<staging>{%c1}
  // CHECK-NEXT: stream.yield %[[TRANSFER_D2H]]
  // CHECK: %[[READY_D2H:.+]] = stream.timepoint.await %[[TIMEPOINT_D2H]] => %[[RESULT_D2H]] : !stream.resource<staging>{%c1}
  // CHECK: %[[LOAD:.+]] = stream.async.load %[[READY_D2H]]
  %2 = stream.async.load %1[%c0] : !stream.resource<staging>{%c1} -> i8
  // CHECK: %[[ADD:.+]] = arith.addi %[[LOAD]], %[[LOAD]]
  %3 = arith.addi %2, %2 : i8
  // CHECK: %[[STORE:.+]] = stream.async.store %[[ADD]], %[[READY_D2H]]
  %4 = stream.async.store %3, %1[%c0] : i8 -> !stream.resource<staging>{%c1}
  // CHECK: %[[RESULT_H2D:.+]], %[[TIMEPOINT_H2D:.+]] = stream.async.execute
  // CHECK-SAME: with(%[[STORE]] as %[[STORE_CAPTURE:.+]]: !stream.resource<staging>{%c1})
  // CHECK-SAME: -> !stream.resource<transient>{%c1}
  // CHECK-NEXT: %[[TRANSFER_H2D:.+]] = stream.async.transfer %[[STORE_CAPTURE]]
  %5 = stream.async.transfer %4 : !stream.resource<staging>{%c1} -> !stream.resource<transient>{%c1}
  // CHECK-NEXT: stream.yield %[[TRANSFER_H2D]]
  // CHECK: %[[READY_H2D:.+]] = stream.timepoint.await %[[TIMEPOINT_H2D]] => %[[RESULT_H2D]] : !stream.resource<transient>{%c1}
  // CHECK: return %[[READY_H2D]]
  return %5 : !stream.resource<transient>
}

// -----

// Tests that partitioning does not hoist ops across cf.asserts.

// CHECK-LABEL: @dontHoistPastAsserts
func.func @dontHoistPastAsserts(%arg0: !stream.resource<external>, %arg1: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %c80 = arith.constant 80 : index
  %c1280 = arith.constant 1280 : index
  %c255_i32 = arith.constant 255 : i32
  %cond_a = arith.constant 0 : i1
  %cond_b = arith.constant 0 : i1

  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.splat
  %2 = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%c1280}
  // CHECK-NEXT: stream.async.dispatch @ex::@dispatch_0
  %3 = stream.async.dispatch @ex::@dispatch_0[%c1, %c1, %c1](%2[%c0 to %c1280 for %c1280], %arg1[%c0 to %c80 for %c80]) : (!stream.resource<transient>{%c1280}, !stream.resource<external>{%c80}) -> %2{%c1280}

  // CHECK: "assert A"
  cf.assert %cond_a, "assert A"

  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.splat
  %4 = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%c20}
  // CHECK-NEXT: stream.async.dispatch @ex::@dispatch_1
  %5 = stream.async.dispatch @ex::@dispatch_1[%c1, %c1, %c1](%arg0[%c0 to %c20 for %c20], %4[%c0 to %c20 for %c20]) : (!stream.resource<external>{%c20}, !stream.resource<transient>{%c20}) -> %4{%c20}

  // CHECK: "assert B"
  cf.assert %cond_b, "assert B"

  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.dispatch @ex::@dispatch_2
  %6 = stream.async.dispatch @ex::@dispatch_2[%c1, %c1, %c1](%3[%c0 to %c1280 for %c1280], %5[%c0 to %c20 for %c20]) : (!stream.resource<transient>{%c1280}, !stream.resource<transient>{%c20}) -> !stream.resource<external>{%c20}

  return %6 : !stream.resource<external>
}

// -----

// Tests that cloning across partition boundaries inserts the cloned op into the
// correct partitions. If the resource is used outside of any partition one of
// the cloned values will be exported to provide the value.

// CHECK-LABEL: @cloneAcrossPartitions
func.func @cloneAcrossPartitions(%cond: i1) -> (!stream.resource<external>, !stream.resource<transient>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c123_i8 = arith.constant 123 : i8

  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.splat
  %splat = stream.async.splat %c123_i8 : i8 -> !stream.resource<transient>{%c1}
  // CHECK-NEXT: stream.async.dispatch
  %dispatch0 = stream.async.dispatch @ex::@dispatch0[%c1, %c1, %c1](%splat[%c0 to %c1 for %c1]) : (!stream.resource<transient>{%c1}) -> !stream.resource<transient>{%c1}
  // CHECK-NEXT: stream.async.transfer
  %download = stream.async.transfer %dispatch0 : !stream.resource<transient>{%c1} -> !stream.resource<staging>{%c1}
  // CHECK: %[[PARTITION0:.+]]:2 = stream.timepoint.await

  // CHECK: stream.async.load
  %load = stream.async.load %download[%c0] : !stream.resource<staging>{%c1} -> i8
  // CHECK: stream.async.store
  %updated = stream.async.store %load, %download[%c0] : i8 -> !stream.resource<staging>{%c1}

  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.transfer
  // CHECK-NEXT: stream.async.splat
  %upload = stream.async.transfer %updated : !stream.resource<staging>{%c1} -> !stream.resource<transient>{%c1}
  // CHECK-NEXT: stream.async.dispatch
  %dispatch1 = stream.async.dispatch @ex::@dispatch1[%c1, %c1, %c1](%upload[%c0 to %c1 for %c1], %splat[%c0 to %c1 for %c1]) : (!stream.resource<transient>{%c1}, !stream.resource<transient>{%c1}) -> !stream.resource<transient>{%c1}
  // CHECK-NEXT: stream.async.transfer
  %result = stream.async.transfer %dispatch1 : !stream.resource<transient>{%c1} -> !stream.resource<external>{%c1}
  // CHECK: %[[PARTITION1:.+]] = stream.timepoint.await

  // CHECK: return %[[PARTITION1]], %[[PARTITION0]]#1
  return %result, %splat : !stream.resource<external>, !stream.resource<transient>
}

// -----

// Tests multiple partitions with dependencies that cross both host and
// device boundaries. Here %1 is used in both partitions and indirectly through
// the arith.select op that executes on the host. In the scheduling code this requires
// tracking both the host and device hazards correctly.

// CHECK-LABEL: @deviceHostDeviceCrossing
func.func @deviceHostDeviceCrossing(%arg0: i1) -> !stream.resource<transient> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c255_i32 = arith.constant 255 : i32

  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.splat
  %0 = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%c128}
  // CHECK-NEXT: stream.async.dispatch @ex::@dispatch0
  %1 = stream.async.dispatch @ex::@dispatch0[%c1, %c1, %c1](%0[%c0 to %c128 for %c128]) : (!stream.resource<transient>{%c128}) -> !stream.resource<transient>{%c128}
  // CHECK-NEXT: stream.async.dispatch @ex::@dispatch1
  %2 = stream.async.dispatch @ex::@dispatch1[%c1, %c1, %c1](%1[%c0 to %c128 for %c128]) : (!stream.resource<transient>{%c128}) -> !stream.resource<transient>{%c128}

  // CHECK: arith.select
  %3 = arith.select %arg0, %1, %2 : !stream.resource<transient>

  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.dispatch @ex::@dispatch2
  %4 = stream.async.dispatch @ex::@dispatch2[%c1, %c1, %c1](%1[%c0 to %c128 for %c128], %3[%c0 to %c128 for %c128]) : (!stream.resource<transient>{%c128}, !stream.resource<transient>{%c128}) -> !stream.resource<transient>{%c128}

  // CHECK: return
  return %4 : !stream.resource<transient>
}

// -----

// Tests that async calls that return only resource types are handled correctly.

stream.async.func private @inplaceExtern(%arg0: !stream.resource<*>, %arg1: index) -> %arg0

// CHECK-LABEL: @inplaceCall
func.func @inplaceCall(%arg0: !stream.resource<*>, %arg1: index, %arg2: index) -> (!stream.resource<*>, index) {
  %c0 = arith.constant 0 : index
  // CHECK: stream.async.execute
  // CHECK-NEXT: stream.async.call
  %0 = stream.async.call @inplaceExtern(%arg0[%c0 to %arg1 for %arg1], %arg2) : (!stream.resource<*>{%arg1}, index) -> %arg0{%arg1}
  // CHECK: stream.timepoint.await
  return %0, %arg1 : !stream.resource<*>, index
}

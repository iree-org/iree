// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-stream-schedule-execution))" %s | FileCheck %s

// Tests that timeline-aware ops don't get immediate awaits.
// The TimelineAwareOpInterface allows ops to participate in timeline scheduling.

// CHECK-LABEL: @timelineAwareCall
util.func public @timelineAwareCall(%resource: !stream.resource<external>, %signal_fence: !stream.test.fence) -> !stream.resource<external> {
  %c16384 = arith.constant 16384 : index

  // Create async execution that produces a timepoint.
  // CHECK: %[[RESULT:.+]], %[[TIMEPOINT:.+]] = stream.async.execute
  %result, %timepoint = stream.async.execute with(%resource as %arg0: !stream.resource<external>{%c16384}) -> !stream.resource<external>{%c16384} {
    %clone = stream.async.clone %arg0 : !stream.resource<external>{%c16384} -> !stream.resource<external>{%c16384}
    stream.yield %clone : !stream.resource<external>{%c16384}
  } => !stream.timepoint

  // Export timepoint to fence for timeline-aware op.
  // CHECK-NOT: stream.timepoint.await %[[TIMEPOINT]]
  // CHECK: %[[WAIT_FENCE:.+]] = stream.timepoint.export %[[TIMEPOINT]]
  %wait_fence = stream.timepoint.export %timepoint => (!stream.test.fence)

  // Call timeline-aware op - should NOT have await before it.
  // CHECK: stream.test.timeline_aware(%[[RESULT]]) waits(%[[WAIT_FENCE]]) signals(%{{.+}})
  %computed = stream.test.timeline_aware(%result) waits(%wait_fence) signals(%signal_fence) : (!stream.resource<external>) -> !stream.resource<external>

  // Import signal fence back to timepoint.
  // CHECK: %[[IMPORTED_TP:.+]] = stream.timepoint.import
  %imported_tp = stream.timepoint.import %signal_fence : (!stream.test.fence) => !stream.timepoint

  // Subsequent async.execute should await imported timepoint.
  // CHECK: %[[FINAL:.+]], %{{.+}} = stream.async.execute await(%[[IMPORTED_TP]])
  %awaited = stream.timepoint.await %imported_tp => %computed : !stream.resource<external>{%c16384}
  %final, %final_tp = stream.async.execute with(%awaited as %arg0: !stream.resource<external>{%c16384}) -> !stream.resource<external>{%c16384} {
    %clone = stream.async.clone %arg0 : !stream.resource<external>{%c16384} -> !stream.resource<external>{%c16384}
    stream.yield %clone : !stream.resource<external>{%c16384}
  } => !stream.timepoint

  // CHECK: util.return %[[FINAL]]
  util.return %final : !stream.resource<external>
}

// -----

// Tests timeline awareness between multiple sequential timeline-aware calls.

// CHECK-LABEL: @sequentialTimelineAwareCalls
util.func public @sequentialTimelineAwareCalls(%resource: !stream.resource<external>, %fence_a: !stream.test.fence, %fence_b: !stream.test.fence) -> !stream.resource<external> {
  %c16384 = arith.constant 16384 : index

  // First async region.
  // CHECK: %[[R1:.+]], %[[T1:.+]] = stream.async.execute
  %r1, %t1 = stream.async.execute with(%resource as %arg0: !stream.resource<external>{%c16384}) -> !stream.resource<external>{%c16384} {
    %clone = stream.async.clone %arg0 : !stream.resource<external>{%c16384} -> !stream.resource<external>{%c16384}
    stream.yield %clone : !stream.resource<external>{%c16384}
  } => !stream.timepoint

  // First timeline-aware call - should NOT have await before it.
  // CHECK-NOT: stream.timepoint.await %[[T1]]
  // CHECK: %[[WAIT_A:.+]] = stream.timepoint.export %[[T1]]
  %wait_a = stream.timepoint.export %t1 => (!stream.test.fence)
  // CHECK: %[[RESULT_A:.+]] = stream.test.timeline_aware(%[[R1]]) waits(%[[WAIT_A]]) signals(%{{.+}})
  %result_a = stream.test.timeline_aware(%r1) waits(%wait_a) signals(%fence_a) : (!stream.resource<external>) -> !stream.resource<external>

  // Import result from first call.
  // CHECK: %[[TP_A:.+]] = stream.timepoint.import
  %tp_a = stream.timepoint.import %fence_a : (!stream.test.fence) => !stream.timepoint

  // Second async region - pass optimizes await into async.execute.
  // CHECK: %[[R2:.+]], %[[T2:.+]] = stream.async.execute await(%[[TP_A]]) => with(%[[RESULT_A]] as %{{.+}}: !stream.resource<external>{%c16384})
  %awaited_a = stream.timepoint.await %tp_a => %result_a : !stream.resource<external>{%c16384}
  %r2, %t2 = stream.async.execute with(%awaited_a as %arg0: !stream.resource<external>{%c16384}) -> !stream.resource<external>{%c16384} {
    %clone = stream.async.clone %arg0 : !stream.resource<external>{%c16384} -> !stream.resource<external>{%c16384}
    stream.yield %clone : !stream.resource<external>{%c16384}
  } => !stream.timepoint

  // Second timeline-aware call - again, no await before.
  // CHECK-NOT: stream.timepoint.await %[[T2]]
  // CHECK: %[[WAIT_B:.+]] = stream.timepoint.export %[[T2]]
  %wait_b = stream.timepoint.export %t2 => (!stream.test.fence)
  // CHECK: %[[RESULT_B:.+]] = stream.test.timeline_aware(%[[R2]]) waits(%[[WAIT_B]]) signals(%{{.+}})
  %result_b = stream.test.timeline_aware(%r2) waits(%wait_b) signals(%fence_b) : (!stream.resource<external>) -> !stream.resource<external>

  // Import and wait for second call.
  // CHECK: %[[TP_B:.+]] = stream.timepoint.import
  %tp_b = stream.timepoint.import %fence_b : (!stream.test.fence) => !stream.timepoint
  // CHECK: %[[AWAITED_B:.+]] = stream.timepoint.await %[[TP_B]] => %[[RESULT_B]]
  %awaited_b = stream.timepoint.await %tp_b => %result_b : !stream.resource<external>{%c16384}

  // CHECK: util.return %[[AWAITED_B]]
  util.return %awaited_b : !stream.resource<external>
}

// -----

// Tests that non-timeline-aware ops still get normal awaits.

util.func private @normal_func(!stream.resource<external>) -> !stream.resource<external>

// CHECK-LABEL: @nonTimelineAwareCall
util.func public @nonTimelineAwareCall(%resource: !stream.resource<external>) -> !stream.resource<external> {
  %c16384 = arith.constant 16384 : index

  // Async work.
  // CHECK: %[[RESULT:.+]], %[[TIMEPOINT:.+]] = stream.async.execute
  %result, %timepoint = stream.async.execute with(%resource as %arg0: !stream.resource<external>{%c16384}) -> !stream.resource<external>{%c16384} {
    %clone = stream.async.clone %arg0 : !stream.resource<external>{%c16384} -> !stream.resource<external>{%c16384}
    stream.yield %clone : !stream.resource<external>{%c16384}
  } => !stream.timepoint

  // Normal call without timeline awareness - SHOULD have await.
  // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[TIMEPOINT]] => %[[RESULT]]
  %awaited = stream.timepoint.await %timepoint => %result : !stream.resource<external>{%c16384}

  // CHECK: util.call @normal_func(%[[AWAITED]])
  %called = util.call @normal_func(%awaited) : (!stream.resource<external>) -> !stream.resource<external>

  util.return %called : !stream.resource<external>
}

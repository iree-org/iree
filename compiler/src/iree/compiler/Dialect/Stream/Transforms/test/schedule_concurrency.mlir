// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module( util.func(iree-stream-schedule-concurrency))" %s | FileCheck %s

// Tests that when favor=min-peak-memory we assume ops are in an order that
// reduces live memory ranges and only optimistically put them in concurrency
// regions when it wouldn't increase the ranges.

// CHECK-LABEL: @partitioningForMinPeakMemory
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<external>, %[[ARG1:.+]]: !stream.resource<external>)
util.func public @partitioningForMinPeakMemory(%arg0: !stream.resource<external>, %arg1: !stream.resource<external>) -> !stream.resource<external>
    attributes {stream.partitioning = #stream.partitioning_config<"min-peak-memory">} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %c80 = arith.constant 80 : index
  %c1280 = arith.constant 1280 : index
  %c255_i32 = arith.constant 255 : i32
  // CHECK: stream.async.execute
  %results, %result_timepoint = stream.async.execute
      // CHECK-SAME: with(%[[ARG1]] as %[[ARG1_CAPTURE:.+]]: !stream.resource<external>{%c80},
      // CHECK-SAME:      %[[ARG0]] as %[[ARG0_CAPTURE:.+]]: !stream.resource<external>{%c20})
      with(%arg1 as %arg2: !stream.resource<external>{%c80},
           %arg0 as %arg3: !stream.resource<external>{%c20})
      -> !stream.resource<external>{%c20} {

    // CHECK: %[[SPLAT0:.+]] = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%c1280}
    %1 = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%c1280}

    // CHECK: %[[CON0:.+]]:2 = stream.async.concurrent
    // CHECK-SAME: with(%[[SPLAT0]] as %[[SPLAT0_CAPTURE:.+]]: !stream.resource<transient>{%c1280},
    // CHECK-SAME:      %[[ARG1_CAPTURE]] as %[[ARG1_CON0_CAPTURE:.+]]: !stream.resource<external>{%c80})
    // CHECK-SAME: -> (!stream.resource<transient>{%c1280}, !stream.resource<transient>{%c20}) {
    // CHECK-NEXT: %[[DISPATCH0:.+]] = stream.async.dispatch @ex::@dispatch_0[%c1, %c1, %c1](%[[SPLAT0_CAPTURE]][{{.+}}], %[[ARG1_CON0_CAPTURE]][{{.+}}])
    %2 = stream.async.dispatch @ex::@dispatch_0[%c1, %c1, %c1](%1[%c0 to %c1280 for %c1280], %arg2[%c0 to %c80 for %c80]) : (!stream.resource<transient>{%c1280}, !stream.resource<external>{%c80}) -> %1{%c1280}
    // CHECK-NEXT: %[[SPLAT1:.+]] = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%c20}
    %3 = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%c20}
    // CHECK-NEXT: stream.yield %[[DISPATCH0]], %[[SPLAT1]] : !stream.resource<transient>{%c1280}, !stream.resource<transient>{%c20}

    // CHECK: %[[DISPATCH1:.+]] = stream.async.dispatch @ex::@dispatch_1[%c1, %c1, %c1](%[[ARG0_CAPTURE]][{{.+}}], %[[CON0]]#1[{{.+}}])
    %4 = stream.async.dispatch @ex::@dispatch_1[%c1, %c1, %c1](%arg3[%c0 to %c20 for %c20], %3[%c0 to %c20 for %c20]) : (!stream.resource<external>{%c20}, !stream.resource<transient>{%c20}) -> %3{%c20}

    // CHECK: %[[DISPATCH2:.+]] = stream.async.dispatch @ex::@dispatch_2[%c1, %c1, %c1](%[[CON0]]#0[{{.+}}], %[[DISPATCH1]][{{.+}}])
    %5 = stream.async.dispatch @ex::@dispatch_2[%c1, %c1, %c1](%2[%c0 to %c1280 for %c1280], %4[%c0 to %c20 for %c20]) : (!stream.resource<transient>{%c1280}, !stream.resource<transient>{%c20}) -> !stream.resource<external>{%c20}

    // CHECK-NEXT: stream.yield %[[DISPATCH2]]
    stream.yield %5 : !stream.resource<external>{%c20}
  } => !stream.timepoint
  %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c20}
  util.return %0 : !stream.resource<external>
}

// -----

// Tests that when favor=max-concurrency we reorder ops aggressively to maximize
// the amount of work scheduled concurrently.

// CHECK-LABEL: @partitioningForMaxConcurrency
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<external>, %[[ARG1:.+]]: !stream.resource<external>)
util.func public @partitioningForMaxConcurrency(%arg0: !stream.resource<external>, %arg1: !stream.resource<external>) -> !stream.resource<external>
    attributes {stream.partitioning = #stream.partitioning_config<"max-concurrency">} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %c80 = arith.constant 80 : index
  %c1280 = arith.constant 1280 : index
  %c255_i32 = arith.constant 255 : i32
  // CHECK: stream.async.execute
  %results, %result_timepoint = stream.async.execute
      // CHECK-SAME: with(%[[ARG1]] as %[[ARG1_CAPTURE:.+]]: !stream.resource<external>{%c80},
      // CHECK-SAME:      %[[ARG0]] as %[[ARG0_CAPTURE:.+]]: !stream.resource<external>{%c20})
      with(%arg1 as %arg2: !stream.resource<external>{%c80},
           %arg0 as %arg3: !stream.resource<external>{%c20})
      -> !stream.resource<external>{%c20} {

    // CHECK: %[[CON0:.+]]:2 = stream.async.concurrent with()
    // CHECK-SAME: -> (!stream.resource<transient>{%c1280}, !stream.resource<transient>{%c20}) {
    // CHECK-NEXT: %[[SPLAT0:.+]] = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%c1280}
    // CHECK-NEXT: %[[SPLAT1:.+]] = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%c20}
    // CHECK-NEXT: stream.yield %[[SPLAT0]], %[[SPLAT1]] : !stream.resource<transient>{%c1280}, !stream.resource<transient>{%c20}

    // CHECK: %[[CON1:.+]]:2 = stream.async.concurrent
    // CHECK-SAME: with(%[[CON0]]#0 as %[[CON0_0_CAPTURE:.+]]: !stream.resource<transient>{%c1280},
    // CHECK-SAME:      %[[ARG1_CAPTURE]] as %[[ARG1_CON1_CAPTURE:.+]]: !stream.resource<external>{%c80},
    // CHECK-SAME:      %[[ARG0_CAPTURE]] as %[[ARG0_CON1_CAPTURE:.+]]: !stream.resource<external>{%c20},
    // CHECK-SAME:      %[[CON0]]#1 as %[[CON0_1_CAPTURE:.+]]: !stream.resource<transient>{%c20})
    // CHECK-SAME: -> (!stream.resource<transient>{%c1280}, !stream.resource<transient>{%c20}) {
    // CHECK-NEXT: %[[DISPATCH0:.+]] = stream.async.dispatch @ex::@dispatch_0[%c1, %c1, %c1](%[[CON0_0_CAPTURE]][{{.+}}], %[[ARG1_CON1_CAPTURE]][{{.+}}])
    // CHECK-NEXT: %[[DISPATCH1:.+]] = stream.async.dispatch @ex::@dispatch_1[%c1, %c1, %c1](%[[ARG0_CON1_CAPTURE]][{{.+}}], %[[CON0_1_CAPTURE]][{{.+}}])
    // CHECK-NEXT: stream.yield %[[DISPATCH0]], %[[DISPATCH1]]

    // CHECK: %[[DISPATCH2:.+]] = stream.async.dispatch @ex::@dispatch_2[%c1, %c1, %c1](%[[CON1]]#0[{{.+}}], %[[CON1]]#1[{{.+}}])
    // CHECK-NEXT: stream.yield %[[DISPATCH2]]

    %1 = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%c1280}
    %2 = stream.async.dispatch @ex::@dispatch_0[%c1, %c1, %c1](%1[%c0 to %c1280 for %c1280], %arg2[%c0 to %c80 for %c80]) : (!stream.resource<transient>{%c1280}, !stream.resource<external>{%c80}) -> %1{%c1280}
    %3 = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%c20}
    %4 = stream.async.dispatch @ex::@dispatch_1[%c1, %c1, %c1](%arg3[%c0 to %c20 for %c20], %3[%c0 to %c20 for %c20]) : (!stream.resource<external>{%c20}, !stream.resource<transient>{%c20}) -> %3{%c20}
    %5 = stream.async.dispatch @ex::@dispatch_2[%c1, %c1, %c1](%2[%c0 to %c1280 for %c1280], %4[%c0 to %c20 for %c20]) : (!stream.resource<transient>{%c1280}, !stream.resource<transient>{%c20}) -> !stream.resource<external>{%c20}
    stream.yield %5 : !stream.resource<external>{%c20}
  } => !stream.timepoint
  %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c20}
  util.return %0 : !stream.resource<external>
}

// -----

// Tests that tied operands properly trigger hazard detection.
// Here @dispatch_1 has a read/write hazard on %capture0 with @dispatch_0 and
// should not be placed into the same concurrency group.

// CHECK-LABEL: @keepTiedOpsSeparate
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<external>)
util.func public @keepTiedOpsSeparate(%arg0: !stream.resource<external>) -> (!stream.resource<external>, !stream.resource<external>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  // CHECK: stream.async.execute
  // CHECK-SAME: with(%[[ARG0]] as %[[CAPTURE0:.+]]: !stream.resource<external>{%c4}) ->
  %results:2, %result_timepoint = stream.async.execute with(%arg0 as %capture0: !stream.resource<external>{%c4}) -> (!stream.resource<external>{%c4}, %arg0{%c4}) {
    // CHECK-NOT: stream.async.concurrent
    // CHECK-NEXT: stream.async.dispatch @ex::@dispatch_0
    %1 = stream.async.dispatch @ex::@dispatch_0(%capture0[%c0 to %c4 for %c4]) : (!stream.resource<external>{%c4}) -> !stream.resource<external>{%c4}
    // CHECK-NEXT: stream.async.dispatch @ex::@dispatch_1
    %2 = stream.async.dispatch @ex::@dispatch_1(%capture0[%c0 to %c4 for %c4]) : (!stream.resource<external>{%c4}) -> %capture0{%c4}
    // CHECK-NEXT: stream.yield
    stream.yield %1, %2 : !stream.resource<external>{%c4}, !stream.resource<external>{%c4}
  } => !stream.timepoint
  util.return %results#0, %results#1 : !stream.resource<external>, !stream.resource<external>
}

// -----

// TODO(#11249): add a test for in-place collectives (send == recv).

// Tests that multiple collective ops will get grouped together in a concurrent
// region.

// CHECK-LABEL: @groupCollectiveOps
// CHECK-SAME: (%[[CHANNEL:.+]]: !stream.channel,
// CHECK-SAME:  %[[SEND0:.+]]: !stream.resource<external>, %[[SEND0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[SEND1:.+]]: !stream.resource<transient>, %[[SEND1_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[RECV_SIZE:[a-z0-9]+]]: index, %[[COUNT:[a-z0-9]+]]: index)
util.func public @groupCollectiveOps(%channel: !stream.channel, %send0: !stream.resource<external>, %send0_size: index, %send1: !stream.resource<transient>, %send1_size: index, %recv_size: index, %count: index) {
  %c0 = arith.constant 0 : index
  // CHECK: stream.async.execute
  %result:2, %result_timepoint = stream.async.execute
      // CHECK-SAME: with(%[[SEND0]] as %[[SEND0_CAPTURE:.+]]: !stream.resource<external>{%[[SEND0_SIZE]]},
      // CHECK-SAME:      %[[SEND1]] as %[[SEND1_CAPTURE:.+]]: !stream.resource<transient>{%[[SEND1_SIZE]]})
      with(%send0 as %captured_send0: !stream.resource<external>{%send0_size},
           %send1 as %captured_send1: !stream.resource<transient>{%send1_size}) ->
           (!stream.resource<transient>{%recv_size}, !stream.resource<transient>{%recv_size}) {

    // CHECK: %[[RECV0:.+]] = stream.async.alloca : !stream.resource<transient>{%[[RECV_SIZE]]}
    // CHECK: %[[RECV1:.+]] = stream.async.alloca : !stream.resource<transient>{%[[RECV_SIZE]]}

    // CHECK: %[[CONCURRENT:.+]]:2 = stream.async.concurrent
    // CHECK-SAME: with(%[[RECV0]] as %[[RECV0_CON_CAPTURE:[a-z0-9]+]]:
    // CHECK-SAME:      %[[SEND0_CAPTURE]] as %[[SEND0_CON_CAPTURE:[a-z0-9]+]]:
    // CHECK-SAME:      %[[RECV1]] as %[[RECV1_CON_CAPTURE:[a-z0-9]+]]:
    // CHECK-SAME:      %[[SEND1_CAPTURE]] as %[[SEND1_CON_CAPTURE:[a-z0-9]+]]:

    // CHECK: %[[CON_CCL0:.+]] = stream.async.collective<all_gather : f32>[%[[COUNT]]]
    // CHECK-SAME: %[[SEND0_CON_CAPTURE]][%c0 to %[[SEND0_SIZE]] for %[[SEND0_SIZE]]],
    // CHECK-SAME: %[[RECV0_CON_CAPTURE]][%c0 to %[[RECV_SIZE]] for %[[RECV_SIZE]]]

    // CHECK: %[[CON_CCL1:.+]] = stream.async.collective<all_gather : f32>[%[[COUNT]]]
    // CHECK-SAME: %[[SEND1_CON_CAPTURE]][%c0 to %[[SEND1_SIZE]] for %[[SEND1_SIZE]]],
    // CHECK-SAME: %[[RECV1_CON_CAPTURE]][%c0 to %[[RECV_SIZE]] for %[[RECV_SIZE]]]

    // CHECK: stream.yield %[[CON_CCL0]], %[[CON_CCL1]]

    %recv0 = stream.async.alloca : !stream.resource<transient>{%recv_size}
    %0 = stream.async.collective<all_gather : f32>[%count] channel(%channel)
        %captured_send0[%c0 to %send0_size for %send0_size],
        %recv0[%c0 to %recv_size for %recv_size] :
        !stream.resource<external>{%send0_size} -> %recv0 as !stream.resource<transient>{%recv_size}

    %recv1 = stream.async.alloca : !stream.resource<transient>{%recv_size}
    %1 = stream.async.collective<all_gather : f32>[%count] channel(%channel)
        %captured_send1[%c0 to %send1_size for %send1_size],
        %recv1[%c0 to %recv_size for %recv_size] :
        !stream.resource<transient>{%send1_size} -> %recv1 as !stream.resource<transient>{%recv_size}

    // CHECK: stream.yield %[[CONCURRENT]]#0, %[[CONCURRENT]]#1
    stream.yield %0, %1 : !stream.resource<transient>{%recv_size}, !stream.resource<transient>{%recv_size}
  } => !stream.timepoint
  util.optimization_barrier %result#0 : !stream.resource<transient>
  util.optimization_barrier %result#1 : !stream.resource<transient>
  util.return
}

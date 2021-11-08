// RUN: iree-opt -split-input-file -pass-pipeline="builtin.func(iree-stream-schedule-concurrency)" %s | IreeFileCheck %s

// CHECK-LABEL: @partitioning
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<external>, %[[ARG1:.+]]: !stream.resource<external>)
func @partitioning(%arg0: !stream.resource<external>, %arg1: !stream.resource<external>) -> !stream.resource<external> {
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %c80 = arith.constant 80 : index
  %c1280 = arith.constant 1280 : index
  %cst = arith.constant 0x7F800000 : f32
  // CHECK: stream.async.execute
  %results, %result_timepoint = stream.async.execute
      // CHECK-SAME: with(%[[ARG1]] as %[[ARG1_CAPTURE:.+]]: !stream.resource<external>{%c80},
      // CHECK-SAME:      %[[ARG0]] as %[[ARG0_CAPTURE:.+]]: !stream.resource<external>{%c20})
      with(%arg1 as %arg2: !stream.resource<external>{%c80},
           %arg0 as %arg3: !stream.resource<external>{%c20})
      -> !stream.resource<external>{%c20} {

    // CHECK: %[[CON0:.+]]:2 = stream.async.concurrent with()
    // CHECK-SAME: -> (!stream.resource<transient>{%c1280}, !stream.resource<transient>{%c20}) {
    // CHECK-NEXT: %[[SPLAT0:.+]] = stream.async.splat %cst : f32 -> !stream.resource<transient>{%c1280}
    // CHECK-NEXT: %[[SPLAT1:.+]] = stream.async.splat %cst : f32 -> !stream.resource<transient>{%c20}
    // CHECK-NEXT: stream.yield %[[SPLAT0]], %[[SPLAT1]] : !stream.resource<transient>{%c1280}, !stream.resource<transient>{%c20}

    // CHECK: %[[CON1:.+]]:2 = stream.async.concurrent
    // CHECK-SAME: with(%[[CON0]]#0 as %[[CON0_0_CAPTURE:.+]]: !stream.resource<transient>{%c1280},
    // CHECK-SAME:      %[[ARG1_CAPTURE]] as %[[ARG1_CON1_CAPTURE:.+]]: !stream.resource<external>{%c80},
    // CHECK-SAME:      %[[ARG0_CAPTURE]] as %[[ARG0_CON1_CAPTURE:.+]]: !stream.resource<external>{%c20},
    // CHECK-SAME:      %[[CON0]]#1 as %[[CON0_1_CAPTURE:.+]]: !stream.resource<transient>{%c20})
    // CHECK-SAME: -> (!stream.resource<transient>{%c1280}, !stream.resource<transient>{%c20}) {
    // CHECK-NEXT: %[[DISPATCH0:.+]] = stream.async.dispatch @ex::@dispatch_0[%c1, %c1, %c1](%[[CON0_0_CAPTURE]], %[[ARG1_CON1_CAPTURE]])
    // CHECK-NEXT: %[[DISPATCH1:.+]] = stream.async.dispatch @ex::@dispatch_1[%c1, %c1, %c1](%[[ARG0_CON1_CAPTURE]], %[[CON0_1_CAPTURE]])
    // CHECK-NEXT: stream.yield %[[DISPATCH0]], %[[DISPATCH1]]

    // CHECK: %[[DISPATCH2:.+]] = stream.async.dispatch @ex::@dispatch_2[%c1, %c1, %c1](%[[CON1]]#0, %[[CON1]]#1)
    // CHECK-NEXT: stream.yield %[[DISPATCH2]]

    %1 = stream.async.splat %cst : f32 -> !stream.resource<transient>{%c1280}
    %2 = stream.async.dispatch @ex::@dispatch_0[%c1, %c1, %c1](%1, %arg2) : (!stream.resource<transient>{%c1280}, !stream.resource<external>{%c80}) -> %1{%c1280}
    %3 = stream.async.splat %cst : f32 -> !stream.resource<transient>{%c20}
    %4 = stream.async.dispatch @ex::@dispatch_1[%c1, %c1, %c1](%arg3, %3) : (!stream.resource<external>{%c20}, !stream.resource<transient>{%c20}) -> %3{%c20}
    %5 = stream.async.dispatch @ex::@dispatch_2[%c1, %c1, %c1](%2, %4) : (!stream.resource<transient>{%c1280}, !stream.resource<transient>{%c20}) -> !stream.resource<external>{%c20}
    stream.yield %5 : !stream.resource<external>{%c20}
  } => !stream.timepoint
  %0 = stream.timepoint.await %result_timepoint => %results : !stream.resource<external>{%c20}
  return %0 : !stream.resource<external>
}

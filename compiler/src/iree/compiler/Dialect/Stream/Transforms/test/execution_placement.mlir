// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-stream-execution-placement))" %s | FileCheck %s

// Tests partitioning multi-device execution with barriers and transfers.
// It validates that multi-stream commands are created and run in parallel.

// CHECK-LABEL: util.func public @deviceMultiDeviceSync
util.func public @deviceMultiDeviceSync(%arg0: i1) -> !stream.resource<transient> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c255_i32 = arith.constant 255 : i32

  %0 = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%c128}
  // CHECK: stream.async.dispatch
  %1 = stream.async.dispatch on(#hal.device.affinity<@device0>) @ex::@dispatch0[%c1, %c1, %c1](%0[%c0 to %c128 for %c128]) : (!stream.resource<transient>{%c128}) -> !stream.resource<transient>{%c128}
  %3 = stream.async.barrier %1 : !stream.resource<transient>{%c128}

  // CHECK: stream.async.transfer
  // CHECK-SAME: on(#hal.device.affinity<@device1>)
  %4 = stream.async.transfer %1 : !stream.resource<transient>{%c128} from(#hal.device.affinity<@device0>) -> to(#hal.device.affinity<@device1>) !stream.resource<transient>{%c128}
  // CHECK: stream.async.dispatch
  %2 = stream.async.dispatch on(#hal.device.affinity<@device1>) @ex::@dispatch1[%c1, %c1, %c1](%0[%c0 to %c128 for %c128]) : (!stream.resource<transient>{%c128}) -> !stream.resource<transient>{%c128}
  %5 = stream.async.barrier %2 : !stream.resource<transient>{%c128}

  // CHECK: stream.async.transfer
  // CHECK-SAME: on(#hal.device.affinity<@device0>)
  %6 = stream.async.transfer %2 : !stream.resource<transient>{%c128} from(#hal.device.affinity<@device1>) -> to(#hal.device.affinity<@device0>) !stream.resource<transient>{%c128}
  // CHECK: stream.async.dispatch
  %7 = stream.async.dispatch on(#hal.device.affinity<@device0>) @ex::@dispatch2[%c1, %c1, %c1](%3[%c0 to %c128 for %c128], %6[%c0 to %c128 for %c128]) : (!stream.resource<transient>{%c128}, !stream.resource<transient>{%c128}) -> !stream.resource<transient>{%c128}
  %8 = stream.async.barrier %7 : !stream.resource<transient>{%c128}
  %9 = stream.async.dispatch on(#hal.device.affinity<@device1>) @ex::@dispatch2[%c1, %c1, %c1](%4[%c0 to %c128 for %c128], %5[%c0 to %c128 for %c128]) : (!stream.resource<transient>{%c128}, !stream.resource<transient>{%c128}) -> !stream.resource<transient>{%c128}

  // CHECK: stream.async.transfer
  // CHECK-SAME: on(#hal.device.affinity<@device1>)
  %10 = stream.async.transfer %9 : !stream.resource<transient>{%c128} from(#hal.device.affinity<@device1>) -> to(#hal.device.affinity<@device0>) !stream.resource<transient>{%c128}
  // CHECK: stream.async.dispatch
  %11 = stream.async.dispatch on(#hal.device.affinity<@device0>) @ex::@dispatch2[%c1, %c1, %c1](%8[%c0 to %c128 for %c128], %10[%c0 to %c128 for %c128]) : (!stream.resource<transient>{%c128}, !stream.resource<transient>{%c128}) -> !stream.resource<transient>{%c128}
  util.return %11 : !stream.resource<transient>
}

// -----

// This one simulates how to do multi-device synchronization between
// more than two devices.

// CHECK-LABEL: @deviceTripleSync
util.func public @deviceTripleSync(%arg0: i1) -> (!stream.resource<transient>, !stream.resource<transient>, !stream.resource<transient>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c255_i32 = arith.constant 255 : i32

  %0 = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%c128}
  %1 = stream.async.dispatch on(#hal.device.affinity<@device0>) @ex::@dispatch0[%c1, %c1, %c1](%0[%c0 to %c128 for %c128]) : (!stream.resource<transient>{%c128}) -> !stream.resource<transient>{%c128}
  %2 = stream.async.barrier %1 : !stream.resource<transient>{%c128}

  %3 = stream.async.dispatch on(#hal.device.affinity<@device1>) @ex::@dispatch0[%c1, %c1, %c1](%0[%c0 to %c128 for %c128]) : (!stream.resource<transient>{%c128}) -> !stream.resource<transient>{%c128}

  // CHECK: stream.async.transfer
  // CHECK-SAME: on(#hal.device.affinity<@device1>)
  %4 = stream.async.transfer %3 : !stream.resource<transient>{%c128} from(#hal.device.affinity<@device1>) -> to(#hal.device.affinity<@device0>) !stream.resource<transient>{%c128}
  %5 = stream.async.dispatch on(#hal.device.affinity<@device2>) @ex::@dispatch0[%c1, %c1, %c1](%0[%c0 to %c128 for %c128]) : (!stream.resource<transient>{%c128}) -> !stream.resource<transient>{%c128}

  // CHECK: stream.async.transfer
  // CHECK-SAME: on(#hal.device.affinity<@device2>)
  %6 = stream.async.transfer %5 : !stream.resource<transient>{%c128} from(#hal.device.affinity<@device2>) -> to(#hal.device.affinity<@device0>) !stream.resource<transient>{%c128}
  %7 = stream.async.dispatch on(#hal.device.affinity<@device0>) @ex::@dispatch2[%c1, %c1, %c1](%2[%c0 to %c128 for %c128], %4[%c0 to %c128 for %c128], %6[%c0 to %c128 for %c128]) : (!stream.resource<transient>{%c128}, !stream.resource<transient>{%c128}, !stream.resource<transient>{%c128}) -> !stream.resource<transient>{%c128}
  %8 = stream.async.barrier %7 : !stream.resource<transient>{%c128}
  %11 = stream.async.dispatch on(#hal.device.affinity<@device0>) @ex::@dispatch0[%c1, %c1, %c1](%8[%c0 to %c128 for %c128]) : (!stream.resource<transient>{%c128}) -> !stream.resource<transient>{%c128}

  // CHECK: stream.async.transfer
  // CHECK-SAME: on(#hal.device.affinity<@device1>)
  %9 = stream.async.transfer %7 : !stream.resource<transient>{%c128} from(#hal.device.affinity<@device0>) -> to(#hal.device.affinity<@device1>) !stream.resource<transient>{%c128}
  %12 = stream.async.dispatch on(#hal.device.affinity<@device1>) @ex::@dispatch0[%c1, %c1, %c1](%9[%c0 to %c128 for %c128]) : (!stream.resource<transient>{%c128}) -> !stream.resource<transient>{%c128}

  // CHECK: stream.async.transfer
  // CHECK-SAME: on(#hal.device.affinity<@device2>)
  %10 = stream.async.transfer %7 : !stream.resource<transient>{%c128} from(#hal.device.affinity<@device0>) -> to(#hal.device.affinity<@device2>) !stream.resource<transient>{%c128}
  %13 = stream.async.dispatch on(#hal.device.affinity<@device2>) @ex::@dispatch0[%c1, %c1, %c1](%10[%c0 to %c128 for %c128]) : (!stream.resource<transient>{%c128}) -> !stream.resource<transient>{%c128}
  util.return %11, %12, %13 : !stream.resource<transient>, !stream.resource<transient>, !stream.resource<transient>
}

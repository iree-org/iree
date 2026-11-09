// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-stream-schedule-execution))" --verify-diagnostics %s

// Tests that scheduling fails directly when partition outlining would produce
// cyclic SSA dependencies. The device_b partition needs the loop result, the
// loop needs the splat cloned into the device_a partition, and that partition
// reads the barrier in the device_b partition.

util.global private @device_a : !hal.device
util.global private @device_b : !hal.device

// expected-error @+1 {{failed to schedule execution partitions without cyclic dependencies}}
util.func public @cyclic_partition_dependencies(%arg0: !stream.resource<transient>) -> (!stream.resource<transient>, !stream.resource<transient>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c255_i32 = arith.constant 255 : i32
  %splat = stream.async.splat on(#hal.device.affinity<@device_a>) %c255_i32 : i32 -> !stream.resource<transient>{%c128}
  %loop = scf.for %i = %c0 to %c4 step %c1 iter_args(%iter = %splat) -> (!stream.resource<transient>) {
    scf.yield %arg0 : !stream.resource<transient>
  }
  %dispatch = stream.async.dispatch on(#hal.device.affinity<@device_b>) @ex::@dispatch[%c1](%arg0[%c0 to %c128 for %c128], %loop[%c0 to %c128 for %c128]) : (!stream.resource<transient>{%c128}, !stream.resource<transient>{%c128}) -> !stream.resource<transient>{%c128}
  %barrier_b = stream.async.barrier on(#hal.device.affinity<@device_b>) %arg0 : !stream.resource<transient>{%c128}
  %barrier_a = stream.async.barrier on(#hal.device.affinity<@device_a>) %barrier_b : !stream.resource<transient>{%c128}
  util.return %dispatch, %barrier_a : !stream.resource<transient>, !stream.resource<transient>
}

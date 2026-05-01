// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-stream-schedule-execution))" --verify-diagnostics %s

// Tests that scheduling fails directly when partition outlining would produce
// cyclic SSA dependencies. The key shape is an interleaved device-0/device-1
// stream where a value produced by the final device-0 partition is needed by an
// earlier device-0 partition through the device-1 side path.

module @module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 : !hal.device
  util.global private @__device_1 : !hal.device

  // expected-error @+1 {{failed to schedule execution partitions without cyclic dependencies}}
  util.func public @cyclic_partition_dependencies(
      %input: !stream.resource<external>,
      %constant0: !stream.resource<constant>,
      %constant1: !stream.resource<constant>,
      %fence: !hal.fence) -> !hal.buffer_view {
    %timepoint = stream.timepoint.immediate => !stream.timepoint
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %to_device_1 = stream.async.transfer %input : !stream.resource<external>{%c8} from(#hal.device.affinity<@__device_0>) -> to(#hal.device.affinity<@__device_1>) !stream.resource<transient>{%c8}
    %base = stream.async.dispatch on(#hal.device.affinity<@__device_0>) @ex::@base(%constant0[%c0 to %c512 for %c512], %input[%c0 to %c8 for %c8]) : (!stream.resource<constant>{%c512}, !stream.resource<external>{%c8}) -> !stream.resource<transient>{%c512}
    %barrier0 = stream.async.barrier on(#hal.device.affinity<@__device_0>) %base : !stream.resource<transient>{%c512}
    %barrier1 = stream.async.barrier on(#hal.device.affinity<@__device_0>) %base : !stream.resource<transient>{%c512}
    %device_1_mid = stream.async.dispatch on(#hal.device.affinity<@__device_1>) @ex::@device_1_mid(%to_device_1[%c0 to %c512 for %c512], %base[%c0 to %c512 for %c512]) : (!stream.resource<transient>{%c512}, !stream.resource<transient>{%c512}) -> !stream.resource<transient>{%c512}
    %device_0_mid = stream.async.dispatch @ex::@device_0_mid(%constant1[%c0 to %c1024 for %c1024], %base[%c0 to %c1024 for %c1024], %base[%c0 to %c4 for %c4]) : (!stream.resource<constant>{%c1024}, !stream.resource<transient>{%c1024}, !stream.resource<transient>{%c4}) -> !stream.resource<transient>{%c512}
    %barrier2 = stream.async.barrier on(#hal.device.affinity<@__device_0>) %device_0_mid : !stream.resource<transient>{%c512}
    %from_device_1 = stream.async.transfer %device_1_mid : !stream.resource<transient>{%c512} from(#hal.device.affinity<@__device_1>) -> !stream.resource<transient>{%c512}
    %barrier3 = stream.async.barrier on(#hal.device.affinity<@__device_0>) %from_device_1 : !stream.resource<transient>{%c512}
    %device_0_join = stream.async.dispatch @ex::@device_0_join(%base[%c0 to %c512 for %c512], %barrier3[%c0 to %c512 for %c512]) : (!stream.resource<transient>{%c512}, !stream.resource<transient>{%c512}) -> !stream.resource<transient>{%c512}
    %device_1_tail = stream.async.dispatch on(#hal.device.affinity<@__device_1>) @ex::@device_1_tail(%device_1_mid[%c0 to %c512 for %c512]) : (!stream.resource<transient>{%c512}) -> !stream.resource<transient>{%c1024}
    %device_0_tail = stream.async.dispatch @ex::@device_0_tail(%base[%c0 to %c512 for %c512], %constant1[%c0 to %c512 for %c512], %device_0_join[%c0 to %c4 for %c4]) : (!stream.resource<transient>{%c1024}, !stream.resource<constant>{%c512}, !stream.resource<transient>{%c4}) -> !stream.resource<transient>{%c256}
    %device_1_result = stream.async.dispatch on(#hal.device.affinity<@__device_1>) @ex::@device_1_result(%device_1_tail[%c0 to %c256 for %c256], %constant0[%c0 to %c512 for %c512]) : (!stream.resource<transient>{%c256}, !stream.resource<constant>{%c512}) -> !stream.resource<transient>{%c512}
    %barrier4 = stream.async.barrier on(#hal.device.affinity<@__device_0>) %device_0_tail : !stream.resource<transient>{%c512}
    %result = stream.async.dispatch @ex::@result(%device_0_tail[%c0 to %c512 for %c512], %device_1_result[%c0 to %c512 for %c512]) : (!stream.resource<transient>{%c512}, !stream.resource<transient>{%c512}) -> !stream.resource<external>{%c512}
    stream.timepoint.chain_external %timepoint => (%fence : !hal.fence)
    %buffer_view = stream.tensor.export %result : tensor<1x1x256xf16> in !stream.resource<external>{%c512} -> !hal.buffer_view
    util.return %buffer_view : !hal.buffer_view
  }
}

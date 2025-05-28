// RUN: iree-opt --iree-hal-transformation-pipeline --split-input-file --verify-diagnostics %s | FileCheck %s

module attributes {stream.affinity.default = #hal.device.affinity<@device_a>} {
  util.global private @device_a = #hal.device.target<"local"> : !hal.device
  util.global private @device_b = #hal.device.target<"metal"> : !hal.device
  util.func public @multi_device_mul() -> !stream.resource<external> {
    %c16 = arith.constant 16 : index
    %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>) : !stream.resource<external>{%c16} => !stream.timepoint
    util.return %result : !stream.resource<external>
  }
}

// Check that for local interop we have a mapping persistent buffer flags.
// CHECK: hal.device.queue.alloca
// CHECK-SAME: MappingPersistent

// -----

// Check that we fail if we try to share allocations with incompatible devices.
module attributes {stream.affinity.default = #hal.device.affinity<@device_a>} {
  util.global private @device_a = #hal.device.target<"local"> : !hal.device
  util.global private @device_b = #hal.device.target<"metal"> : !hal.device
  util.global private @device_c = #hal.device.target<"vulkan"> : !hal.device
  util.func public @multi_device_mul() -> !stream.resource<external> {
    %c16 = arith.constant 16 : index
    // expected-error @+1 {{failed to legalize operation 'stream.resource.alloca' that was explicitly marked illegal}}
    %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>, #hal.device.affinity<@device_c>]>) : !stream.resource<external>{%c16} => !stream.timepoint
    util.return %result : !stream.resource<external>
  }
}

// RUN: iree-opt --split-input-file --iree-hal-conversion --cse --iree-hal-indirect-command-buffers=true --iree-hal-indirect-constant-bindings=false %s | FileCheck %s --check-prefixes=CHECK,DIRECT
// RUN: iree-opt --split-input-file --iree-hal-conversion --cse --iree-hal-indirect-command-buffers=true --iree-hal-indirect-constant-bindings=true %s | FileCheck %s --check-prefixes=CHECK,INDIRECT

// Tests that constants are referenced through the binding table on devices
// with `indirect_constant_bindings = true` and on all devices with
// --iree-hal-indirect-constant-bindings. Each region has two copies so it is
// recorded into a command buffer instead of becoming a single queue copy.

util.global private @device_indirect = #hal.device.target<"local", {indirect_constant_bindings = true}, [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">]> : !hal.device
util.global private @device_direct = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">]> : !hal.device
util.global private @constant_resource : !stream.resource<constant>
util.global private @constant_size : index

// CHECK-LABEL: @cmdCopyConstant
//  CHECK-SAME: (%[[ARG_RESOURCE:.+]]: !hal.buffer)
util.func public @cmdCopyConstant(%arg_resource: !stream.resource<external>) -> (!stream.timepoint, !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %arg_size = arith.constant 200 : index
  // CHECK-DAG: %[[CONSTANT_RESOURCE:.+]] = util.global.load immutable @constant_resource
  %constant_resource = util.global.load immutable @constant_resource : !stream.resource<constant>
  // CHECK-DAG: %[[CONSTANT_SIZE:.+]] = util.global.load immutable @constant_size
  %constant_size = util.global.load immutable @constant_size : index

  // CHECK: %[[DEVICE_INDIRECT:.+]] = util.global.load immutable @device_indirect
  // CHECK: hal.command_buffer.copy_buffer
  // CHECK-SAME: source(%c0 : index)[%c0]
  // CHECK-SAME: target(%c1 : index)[%c0]
  //      CHECK: hal.device.queue.execute.indirect<%[[DEVICE_INDIRECT]]
  // CHECK-SAME: bindings([
  // CHECK-NEXT:   (%[[CONSTANT_RESOURCE]] : !hal.buffer)[%c0, %[[CONSTANT_SIZE]]],
  // CHECK-NEXT:   (%[[ARG_RESOURCE]] : !hal.buffer)[%c0, %c200]
  // CHECK-NEXT: ])
  %0 = stream.cmd.execute on(#hal.device.affinity<@device_indirect>) with(%constant_resource as %constant_capture: !stream.resource<constant>{%constant_size}, %arg_resource as %arg_capture: !stream.resource<external>{%arg_size}) {
    stream.cmd.copy %constant_capture[%c0], %arg_capture[%c0], %c128 : !stream.resource<constant>{%constant_size} -> !stream.resource<external>{%arg_size}
    stream.cmd.copy %constant_capture[%c0], %arg_capture[%c128], %c128 : !stream.resource<constant>{%constant_size} -> !stream.resource<external>{%arg_size}
  } => !stream.timepoint

  // CHECK: %[[DEVICE_DIRECT:.+]] = util.global.load immutable @device_direct
  // CHECK: hal.command_buffer.copy_buffer
  // DIRECT-SAME: source(%[[CONSTANT_RESOURCE]] : !hal.buffer)[%c0]
  // DIRECT-SAME: target(%c0 : index)[%c0]
  // INDIRECT-SAME: source(%c0 : index)[%c0]
  // INDIRECT-SAME: target(%c1 : index)[%c0]
  //      CHECK: hal.device.queue.execute.indirect<%[[DEVICE_DIRECT]]
  // CHECK-SAME: bindings([
  // INDIRECT-NEXT: (%[[CONSTANT_RESOURCE]] : !hal.buffer)[%c0, %[[CONSTANT_SIZE]]],
  // CHECK-NEXT:   (%[[ARG_RESOURCE]] : !hal.buffer)[%c0, %c200]
  // CHECK-NEXT: ])
  %1 = stream.cmd.execute on(#hal.device.affinity<@device_direct>) with(%constant_resource as %constant_capture: !stream.resource<constant>{%constant_size}, %arg_resource as %arg_capture: !stream.resource<external>{%arg_size}) {
    stream.cmd.copy %constant_capture[%c0], %arg_capture[%c0], %c128 : !stream.resource<constant>{%constant_size} -> !stream.resource<external>{%arg_size}
    stream.cmd.copy %constant_capture[%c0], %arg_capture[%c128], %c128 : !stream.resource<constant>{%constant_size} -> !stream.resource<external>{%arg_size}
  } => !stream.timepoint
  util.return %0, %1 : !stream.timepoint, !stream.timepoint
}

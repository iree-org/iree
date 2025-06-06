// RUN: iree-opt --split-input-file --iree-hal-conversion %s | FileCheck %s

util.global private @device : !hal.device

// CHECK-LABEL: @file_constant
//  CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer)
util.func public @file_constant(%buffer: !util.buffer) {
  %c0 = arith.constant 0 : index
  %c1088 = arith.constant 1088 : index
  // CHECK: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK: = hal.ex.file.from_memory device(%[[DEVICE]] : !hal.device) affinity(%c-1_i64) access(Read) buffer(%[[BUFFER]] : !util.buffer)[%c0 for %c1088] flags(%c0_i32) : !hal.file
  %file = stream.file.constant on(#hal.device.affinity<@device>) %buffer[%c0 for %c1088] : !util.buffer{%c1088} -> !stream.file
  util.return
}

// -----

// TODO(multi-device): emit policy ops to select the device. Today the first
// affinity specified is chosen.

util.global private @device_a : !hal.device
util.global private @device_b : !hal.device

// CHECK-LABEL: @file_constant_optimal
//  CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer)
util.func public @file_constant_optimal(%buffer: !util.buffer) {
  %c0 = arith.constant 0 : index
  %c1088 = arith.constant 1088 : index
  // CHECK: %[[DEVICE:.+]] = util.global.load immutable @device_a
  // CHECK: = hal.ex.file.from_memory device(%[[DEVICE]] : !hal.device) affinity(%c-1_i64) access(Read) buffer(%[[BUFFER]] : !util.buffer)[%c0 for %c1088] flags(%c0_i32) : !hal.file
  %file = stream.file.constant on(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>) %buffer[%c0 for %c1088] : !util.buffer{%c1088} -> !stream.file
  util.return
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @file_read
//  CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[FILE:.+]]: !hal.file, %[[RESOURCE:.+]]: !hal.buffer)
util.func public @file_read(%wait: !stream.timepoint, %file: !stream.file, %resource: !stream.resource<variable>) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1088 = arith.constant 1088 : index
  // CHECK: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK: %[[SIGNAL:.+]] = hal.fence.create
  // CHECK: hal.device.queue.read<%[[DEVICE]] : !hal.device> affinity(%c-1_i64) wait(%[[WAIT]]) signal(%[[SIGNAL]]) source(%[[FILE]] : !hal.file)[%c0_i64] target(%[[RESOURCE]] : !hal.buffer)[%c0] length(%c1088) flags("None")
  %signal = stream.file.read on(#hal.device.affinity<@device>) await(%wait) => %file[%c0_i64], %resource[%c0], %c1088 : !stream.file -> !stream.resource<variable>{%c1088} => !stream.timepoint
  // CHECK: util.return %[[SIGNAL]]
  util.return %signal : !stream.timepoint
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @file_write
//  CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[FILE:.+]]: !hal.file, %[[RESOURCE:.+]]: !hal.buffer)
util.func public @file_write(%wait: !stream.timepoint, %file: !stream.file, %resource: !stream.resource<variable>) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1088 = arith.constant 1088 : index
  // CHECK: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK: %[[SIGNAL:.+]] = hal.fence.create
  // CHECK: hal.device.queue.write<%[[DEVICE]] : !hal.device> affinity(%c-1_i64) wait(%[[WAIT]]) signal(%[[SIGNAL]]) source(%[[RESOURCE]] : !hal.buffer)[%c0] target(%[[FILE]] : !hal.file)[%c0_i64] length(%c1088) flags("None")
  %signal = stream.file.write on(#hal.device.affinity<@device>) await(%wait) => %resource[%c0], %file[%c0_i64], %c1088 : !stream.resource<variable>{%c1088} -> !stream.file => !stream.timepoint
  // CHECK: util.return %[[SIGNAL]]
  util.return %signal : !stream.timepoint
}

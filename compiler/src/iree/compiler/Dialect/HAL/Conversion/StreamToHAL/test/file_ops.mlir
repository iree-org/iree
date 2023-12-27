// RUN: iree-opt --split-input-file --iree-hal-conversion %s | FileCheck %s

// CHECK-LABEL: @file_constant
//  CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer)
func.func @file_constant(%buffer: !util.buffer) {
  %c0 = arith.constant 0 : index
  %c1088 = arith.constant 1088 : index
  // CHECK: %[[DEVICE:.+]] = hal.devices.get %{{.+}}
  // CHECK: = hal.ex.file.from_memory device(%[[DEVICE]] : !hal.device) affinity(%c-1_i64) access(Read) buffer(%[[BUFFER]] : !util.buffer)[%c0 for %c1088] flags(%c0_i32) : !hal.file
  %file = stream.file.constant %buffer[%c0 for %c1088] : !util.buffer{%c1088} -> !stream.file
  return
}

// -----

// CHECK-LABEL: @file_read
//  CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[FILE:.+]]: !hal.file, %[[RESOURCE:.+]]: !hal.buffer)
func.func @file_read(%wait: !stream.timepoint, %file: !stream.file, %resource: !stream.resource<variable>) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1088 = arith.constant 1088 : index
  // CHECK: %[[DEVICE:.+]] = hal.devices.get %{{.+}}
  // CHECK: %[[SIGNAL:.+]] = hal.fence.create
  // CHECK: hal.device.queue.read<%[[DEVICE]] : !hal.device> affinity(%c-1_i64) wait(%[[WAIT]]) signal(%[[SIGNAL]]) source(%[[FILE]] : !hal.file)[%c0_i64] target(%[[RESOURCE]] : !hal.buffer)[%c0] length(%c1088) flags(0)
  %signal = stream.file.read await(%wait) => %file[%c0_i64], %resource[%c0], %c1088 : !stream.file -> !stream.resource<variable>{%c1088} => !stream.timepoint
  // CHECK: return %[[SIGNAL]]
  return %signal : !stream.timepoint
}

// -----

// CHECK-LABEL: @file_write
//  CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[FILE:.+]]: !hal.file, %[[RESOURCE:.+]]: !hal.buffer)
func.func @file_write(%wait: !stream.timepoint, %file: !stream.file, %resource: !stream.resource<variable>) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1088 = arith.constant 1088 : index
  // CHECK: %[[DEVICE:.+]] = hal.devices.get %{{.+}}
  // CHECK: %[[SIGNAL:.+]] = hal.fence.create
  // CHECK: hal.device.queue.write<%[[DEVICE]] : !hal.device> affinity(%c-1_i64) wait(%[[WAIT]]) signal(%[[SIGNAL]]) source(%[[RESOURCE]] : !hal.buffer)[%c0] target(%[[FILE]] : !hal.file)[%c0_i64] length(%c1088) flags(0)
  %signal = stream.file.write await(%wait) => %resource[%c0], %file[%c0_i64], %c1088 : !stream.resource<variable>{%c1088} -> !stream.file => !stream.timepoint
  // CHECK: return %[[SIGNAL]]
  return %signal : !stream.timepoint
}

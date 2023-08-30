// RUN: iree-opt --split-input-file --iree-hal-inline-conversion %s | FileCheck %s

// NOTE: file ops are not available under the inline HAL and are all no-oped.
// This works today because file ops are only used as a fallback for direct
// resource importing. If an input program relied on file ops for user behavior
// we'd need to support them and expose them through the runtime HAL module.

// CHECK-LABEL: @file_constant
//  CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer)
func.func @file_constant(%buffer: !util.buffer) {
  %c0 = arith.constant 0 : index
  %c1088 = arith.constant 1088 : index
  // CHECK: = arith.constant 0 : i32
  %file = stream.file.constant %buffer[%c0 for %c1088] : !util.buffer{%c1088} -> !stream.file
  return
}

// -----

// CHECK-LABEL: @file_read
//  CHECK-SAME: (%[[WAIT:.+]]: i64, %[[FILE:.+]]: i32, %[[RESOURCE:.+]]: !util.buffer)
func.func @file_read(%wait: !stream.timepoint, %file: !stream.file, %resource: !stream.resource<variable>) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %offset = arith.constant 100 : i64
  %c1088 = arith.constant 1088 : index
  // CHECK: %[[SIGNAL:.+]] = arith.constant 0 : i64
  %signal = stream.file.read await(%wait) => %file[%offset], %resource[%c0], %c1088 : !stream.file -> !stream.resource<variable>{%c1088} => !stream.timepoint
  // CHECK: return %[[SIGNAL]]
  return %signal : !stream.timepoint
}

// -----

// CHECK-LABEL: @file_write
//  CHECK-SAME: (%[[WAIT:.+]]: i64, %[[FILE:.+]]: i32, %[[RESOURCE:.+]]: !util.buffer)
func.func @file_write(%wait: !stream.timepoint, %file: !stream.file, %resource: !stream.resource<variable>) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %offset = arith.constant 100 : i64
  %c1088 = arith.constant 1088 : index
  // CHECK: %[[SIGNAL:.+]] = arith.constant 0 : i64
  %signal = stream.file.write await(%wait) => %resource[%c0], %file[%offset], %c1088 : !stream.resource<variable>{%c1088} -> !stream.file => !stream.timepoint
  // CHECK: return %[[SIGNAL]]
  return %signal : !stream.timepoint
}

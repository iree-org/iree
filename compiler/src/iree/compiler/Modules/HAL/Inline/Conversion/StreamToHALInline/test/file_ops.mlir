// RUN: iree-opt --split-input-file --iree-hal-inline-conversion %s | FileCheck %s

// NOTE: file ops are not available under the inline HAL and are all no-oped.
// This works today because file ops are only used as a fallback for direct
// resource importing. If an input program relied on file ops for user behavior
// we'd need to support them and expose them through the runtime HAL module.

// CHECK-LABEL: @file_constant
//  CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer) -> !util.buffer
func.func @file_constant(%buffer: !util.buffer) -> !stream.file {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  // CHECK: %[[SPAN:.+]] = util.buffer.subspan %[[BUFFER]][%c100] : !util.buffer{%c300} -> !util.buffer{%c200}
  %file = stream.file.constant %buffer[%c100 for %c200] : !util.buffer{%c300} -> !stream.file
  // CHECK: return %[[SPAN]]
  return %file : !stream.file
}

// -----

// CHECK-LABEL: @file_read
//  CHECK-SAME: (%[[WAIT:.+]]: i64, %[[FILE:.+]]: !util.buffer, %[[RESOURCE:.+]]: !util.buffer)
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
//  CHECK-SAME: (%[[WAIT:.+]]: i64, %[[FILE:.+]]: !util.buffer, %[[RESOURCE:.+]]: !util.buffer)
func.func @file_write(%wait: !stream.timepoint, %file: !stream.file, %resource: !stream.resource<variable>) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %offset = arith.constant 100 : i64
  %c1088 = arith.constant 1088 : index
  // CHECK: %[[FILE_SIZE:.+]] = util.buffer.size %[[FILE]]
  // CHECK: util.buffer.copy %[[RESOURCE]][%c0], %[[FILE]][%c100], %c1088 : !util.buffer{%c1088} -> !util.buffer{%[[FILE_SIZE]]}
  // CHECK: %[[SIGNAL:.+]] = arith.constant 0 : i64
  %signal = stream.file.write await(%wait) => %resource[%c0], %file[%offset], %c1088 : !stream.resource<variable>{%c1088} -> !stream.file => !stream.timepoint
  // CHECK: return %[[SIGNAL]]
  return %signal : !stream.timepoint
}

// -----

// CHECK-LABEL: @variable_read
//  CHECK-SAME: (%[[WAIT:.+]]: i64) -> (!util.buffer, i64)
func.func @variable_read(%wait: !stream.timepoint) -> (!stream.resource<variable>, !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c100 = arith.constant 100 : i64
  // CHECK: %[[CONSTANT:.+]] = util.buffer.constant
  %constant = util.buffer.constant {alignment = 64 : index} : !util.buffer = dense<1> : tensor<64xi8>
  // CHECK: %[[BUFFER:.+]], %[[STORAGE:.+]] = hal_inline.buffer.allocate
  %resource = stream.resource.alloc uninitialized : !stream.resource<variable>{%c64}
  // CHECK: %[[SPAN:.+]] = util.buffer.subspan %[[CONSTANT]][%c16] : !util.buffer{%c64} -> !util.buffer{%c32}
  %file = stream.file.constant %constant[%c16 for %c32] : !util.buffer{%c64} -> !stream.file
  // CHECK: %[[SPAN_SIZE:.+]] = util.buffer.size %[[SPAN]]
  // CHECK: util.buffer.copy %[[SPAN]][%c100], %[[STORAGE]][%c32], %c32 : !util.buffer{%[[SPAN_SIZE]]} -> !util.buffer{%c64}
  // CHECK: %[[SIGNAL:.+]] = arith.constant 0 : i64
  %signal = stream.file.read await(%wait) => %file[%c100], %resource[%c32], %c32 : !stream.file -> !stream.resource<variable>{%c64} => !stream.timepoint
  // CHECK: return %[[STORAGE]], %[[SIGNAL]]
  return %resource, %signal : !stream.resource<variable>, !stream.timepoint
}

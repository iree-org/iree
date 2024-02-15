// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @file_constant
//  CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer)
util.func private @file_constant(%buffer: !util.buffer) {
  %c0 = arith.constant 0 : index
  %c1088 = arith.constant 1088 : index
  // CHECK: %file = stream.file.constant %[[BUFFER]][%c0 for %c1088] : !util.buffer{%c1088} -> !stream.file
  %file = stream.file.constant %buffer[%c0 for %c1088] : !util.buffer{%c1088} -> !stream.file
  util.return
}

// -----

// CHECK-LABEL: @file_read
//  CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[FILE:.+]]: !stream.file, %[[RESOURCE:.+]]: !stream.resource<variable>)
util.func private @file_read(%wait: !stream.timepoint, %file: !stream.file, %resource: !stream.resource<variable>) {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1088 = arith.constant 1088 : index
  // CHECK: = stream.file.read await(%[[WAIT]]) => %[[FILE]][%c0_i64], %[[RESOURCE]][%c0], %c1088 : !stream.file -> !stream.resource<variable>{%c1088} => !stream.timepoint
  %0 = stream.file.read await(%wait) => %file[%c0_i64], %resource[%c0], %c1088 : !stream.file -> !stream.resource<variable>{%c1088} => !stream.timepoint
  util.return
}

// -----

// CHECK-LABEL: @file_write
//  CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[FILE:.+]]: !stream.file, %[[RESOURCE:.+]]: !stream.resource<variable>)
util.func private @file_write(%wait: !stream.timepoint, %file: !stream.file, %resource: !stream.resource<variable>) {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1088 = arith.constant 1088 : index
  // CHECK: = stream.file.write await(%[[WAIT]]) => %[[RESOURCE]][%c0], %[[FILE]][%c0_i64], %c1088 : !stream.resource<variable>{%c1088} -> !stream.file => !stream.timepoint
  %0 = stream.file.write await(%wait) => %resource[%c0], %file[%c0_i64], %c1088 : !stream.resource<variable>{%c1088} -> !stream.file => !stream.timepoint
  util.return
}

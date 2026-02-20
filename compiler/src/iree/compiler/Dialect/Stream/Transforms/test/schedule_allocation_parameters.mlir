// RUN: iree-opt --split-input-file --iree-stream-schedule-allocation %s | FileCheck %s

// Tests that async parameter ops are converted to cmd parameter ops with
// dynamic (!util.buffer) scope and key operands.

// CHECK-LABEL: @parameterLoad
// CHECK-SAME: (%[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer)
util.func public @parameterLoad(%scope: !util.buffer, %key: !util.buffer) -> !stream.resource<constant> {
  %c0_i64 = arith.constant 0 : i64
  %c1024 = arith.constant 1024 : index
  // CHECK: %[[RESULTS:.+]], %[[RESULT_TP:.+]] = stream.cmd.parameter.load
  // CHECK-SAME: {
  // CHECK-NEXT:   %[[SCOPE]]::%[[KEY]]
  // CHECK-SAME:       : !stream.resource<constant>
  // CHECK-NEXT: } =>
  %result, %result_tp = stream.async.parameter.load %scope::%key[%c0_i64]
      : !stream.resource<constant>{%c1024} => !stream.timepoint
  %awaited = stream.timepoint.await %result_tp => %result
      : !stream.resource<constant>{%c1024}
  util.return %awaited : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @parameterLoadNoScope
// CHECK-SAME: (%[[KEY:.+]]: !util.buffer)
util.func public @parameterLoadNoScope(%key: !util.buffer) -> !stream.resource<constant> {
  %c0_i64 = arith.constant 0 : i64
  %c512 = arith.constant 512 : index
  // CHECK: %[[RESULTS:.+]], %[[RESULT_TP:.+]] = stream.cmd.parameter.load
  // CHECK-SAME: {
  // CHECK-NEXT:   %[[KEY]]
  // CHECK-NOT:    ::
  // CHECK-SAME:       : !stream.resource<constant>
  // CHECK-NEXT: } =>
  %result, %result_tp = stream.async.parameter.load %key[%c0_i64]
      : !stream.resource<constant>{%c512} => !stream.timepoint
  %awaited = stream.timepoint.await %result_tp => %result
      : !stream.resource<constant>{%c512}
  util.return %awaited : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @parameterRead
// CHECK-SAME: (%[[TARGET:.+]]: !stream.resource<transient>, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer)
util.func public @parameterRead(%target: !stream.resource<transient>, %scope: !util.buffer, %key: !util.buffer) -> !stream.resource<transient> {
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  // CHECK: stream.cmd.parameter.read
  // CHECK-SAME: %[[SCOPE]]::%[[KEY]]
  // CHECK-SAME: -> %[[TARGET]]
  %result, %result_tp = stream.async.parameter.read
      %scope::%key[%c0_i64] -> %target[%c0 to %c1024 for %c1024]
      : !stream.resource<transient>{%c1024} => !stream.timepoint
  %awaited = stream.timepoint.await %result_tp => %result
      : !stream.resource<transient>{%c1024}
  util.return %awaited : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @parameterWrite
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<transient>, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer)
util.func public @parameterWrite(%source: !stream.resource<transient>, %scope: !util.buffer, %key: !util.buffer) -> !stream.resource<transient> {
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  // CHECK: stream.cmd.parameter.write
  // CHECK-SAME: %[[SOURCE]]
  // CHECK-SAME: -> %[[SCOPE]]::%[[KEY]]
  %result, %result_tp = stream.async.parameter.write
      %source[%c0 to %c1024 for %c1024] -> %scope::%key[%c0_i64]
      : !stream.resource<transient>{%c1024} => !stream.timepoint
  %awaited = stream.timepoint.await %result_tp => %result
      : !stream.resource<transient>{%c1024}
  util.return %awaited : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @parameterWriteNoScope
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<transient>, %[[KEY:.+]]: !util.buffer)
util.func public @parameterWriteNoScope(%source: !stream.resource<transient>, %key: !util.buffer) -> !stream.resource<transient> {
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  // CHECK: stream.cmd.parameter.write
  // CHECK-SAME: %[[SOURCE]]
  // CHECK-SAME: -> %[[KEY]]
  // CHECK-NOT:  ::
  %result, %result_tp = stream.async.parameter.write
      %source[%c0 to %c256 for %c256] -> %key[%c0_i64]
      : !stream.resource<transient>{%c256} => !stream.timepoint
  %awaited = stream.timepoint.await %result_tp => %result
      : !stream.resource<transient>{%c256}
  util.return %awaited : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @parameterGather
// CHECK-SAME: (%[[TARGET:.+]]: !stream.resource<transient>, %[[SCOPE:.+]]: !util.buffer, %[[KEY0:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer)
util.func public @parameterGather(%target: !stream.resource<transient>, %scope: !util.buffer, %key0: !util.buffer, %key1: !util.buffer) -> !stream.resource<transient> {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index
  %c2048 = arith.constant 2048 : index
  // CHECK: stream.cmd.parameter.gather
  // CHECK-SAME: {
  // CHECK-NEXT:   %[[SCOPE]]::%[[KEY0]]
  // CHECK-SAME:   -> %[[TARGET]]
  // CHECK-NEXT:   %[[SCOPE]]::%[[KEY1]]
  // CHECK-SAME:   -> %[[TARGET]]
  // CHECK-NEXT: } =>
  %result, %result_tp = stream.async.parameter.gather {
    %scope::%key0[%c0_i64] -> %target[%c0 to %c512 for %c512] : !stream.resource<transient>{%c2048},
    %scope::%key1[%c1_i64] -> %target[%c512 to %c1024 for %c512] : !stream.resource<transient>{%c2048}
  } : !stream.resource<transient> => !stream.timepoint
  %awaited = stream.timepoint.await %result_tp => %result
      : !stream.resource<transient>{%c2048}
  util.return %awaited : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @parameterScatter
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<transient>, %[[SCOPE:.+]]: !util.buffer, %[[KEY0:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer)
util.func public @parameterScatter(%source: !stream.resource<transient>, %scope: !util.buffer, %key0: !util.buffer, %key1: !util.buffer) -> !stream.resource<transient> {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index
  %c2048 = arith.constant 2048 : index
  // CHECK: stream.cmd.parameter.scatter
  // CHECK-SAME: {
  // CHECK-NEXT:   %[[SOURCE]]
  // CHECK-SAME:   -> %[[SCOPE]]::%[[KEY0]]
  // CHECK-NEXT:   %[[SOURCE]]
  // CHECK-SAME:   -> %[[SCOPE]]::%[[KEY1]]
  // CHECK-NEXT: } =>
  %result, %result_tp = stream.async.parameter.scatter {
    %source[%c0 to %c512 for %c512] : !stream.resource<transient>{%c2048} -> %scope::%key0[%c0_i64],
    %source[%c512 to %c1024 for %c512] : !stream.resource<transient>{%c2048} -> %scope::%key1[%c1_i64]
  } : !stream.resource<transient> => !stream.timepoint
  %awaited = stream.timepoint.await %result_tp => %result
      : !stream.resource<transient>{%c2048}
  util.return %awaited : !stream.resource<transient>
}

// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-stream-elide-timepoints)" %s | FileCheck %s

// Tests that timepoint elision works correctly with parameter load/write ops,
// which are timeline ops that produce and consume timepoints. These tests
// exercise the interaction between parameter ops and the await folding and
// cleanup passes.

// Tests that an await of a parameter load timepoint is absorbed into a
// consuming timeline op.

// CHECK-LABEL: @parameterLoadAwaitFoldsIntoConsumer
// CHECK-SAME: (%[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer)
util.func public @parameterLoadAwaitFoldsIntoConsumer(%scope: !util.buffer, %key: !util.buffer) -> !stream.resource<external> {
  %c0_i64 = arith.constant 0 : i64
  %c1024 = arith.constant 1024 : index

  // Load from parameter (produces resource + timepoint).
  // CHECK: %[[LOADED:.+]], %[[LOAD_TP:.+]] = stream.async.parameter.load
  %loaded, %load_tp = stream.async.parameter.load %scope::%key[%c0_i64]
      : !stream.resource<constant>{%c1024} => !stream.timepoint

  // Await should be eliminated.
  // CHECK-NOT: stream.timepoint.await
  %awaited = stream.timepoint.await %load_tp => %loaded
      : !stream.resource<constant>{%c1024}

  // Consumer should absorb the load's timepoint.
  // CHECK: %[[RESULT:.+]], %{{.+}} = stream.test.timeline_op await(%[[LOAD_TP]]) => with(%[[LOADED]])
  %result, %result_tp = stream.test.timeline_op with(%awaited)
      : (!stream.resource<constant>{%c1024})
      -> !stream.resource<external>{%c1024}
      => !stream.timepoint
  %result_awaited = stream.timepoint.await %result_tp => %result
      : !stream.resource<external>{%c1024}
  util.return %result_awaited : !stream.resource<external>
}

// -----

// Tests that an await of a parameter write's result timepoint is absorbed into
// a consuming timeline op.

// CHECK-LABEL: @parameterWriteAwaitFoldsIntoConsumer
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<external>, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer)
util.func public @parameterWriteAwaitFoldsIntoConsumer(%source: !stream.resource<external>, %scope: !util.buffer, %key: !util.buffer) -> !stream.resource<external> {
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index

  // Write to parameter (returns tied source + timepoint).
  // CHECK: %[[WRITTEN:.+]], %[[WRITE_TP:.+]] = stream.async.parameter.write
  %written, %write_tp = stream.async.parameter.write
      %source[%c0 to %c1024 for %c1024] -> %scope::%key[%c0_i64]
      : !stream.resource<external>{%c1024} => !stream.timepoint

  // Await should be eliminated.
  // CHECK-NOT: stream.timepoint.await
  %awaited = stream.timepoint.await %write_tp => %written
      : !stream.resource<external>{%c1024}

  // Consumer should absorb the write's timepoint.
  // CHECK: %[[RESULT:.+]], %{{.+}} = stream.test.timeline_op await(%[[WRITE_TP]]) => with(%[[WRITTEN]])
  %result, %result_tp = stream.test.timeline_op with(%awaited)
      : (!stream.resource<external>{%c1024})
      -> !stream.resource<external>{%c1024}
      => !stream.timepoint
  %result_awaited = stream.timepoint.await %result_tp => %result
      : !stream.resource<external>{%c1024}
  util.return %result_awaited : !stream.resource<external>
}

// -----

// Tests that two independent parameter loads with their awaits both fold into
// a consuming timeline op, producing a joined timepoint.

// CHECK-LABEL: @twoParameterLoadsAwaitFoldIntoConsumer
// CHECK-SAME: (%[[SCOPE:.+]]: !util.buffer, %[[KEY0:.+]]: !util.buffer, %[[KEY1:.+]]: !util.buffer)
util.func public @twoParameterLoadsAwaitFoldIntoConsumer(%scope: !util.buffer, %key0: !util.buffer, %key1: !util.buffer) -> !stream.resource<external> {
  %c0_i64 = arith.constant 0 : i64
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index

  // Two independent parameter loads.
  // CHECK: %[[LOADED0:.+]], %[[TP0:.+]] = stream.async.parameter.load
  %loaded0, %tp0 = stream.async.parameter.load %scope::%key0[%c0_i64]
      : !stream.resource<constant>{%c512} => !stream.timepoint
  // CHECK: %[[LOADED1:.+]], %[[TP1:.+]] = stream.async.parameter.load
  %loaded1, %tp1 = stream.async.parameter.load %scope::%key1[%c0_i64]
      : !stream.resource<constant>{%c1024} => !stream.timepoint

  // Both awaits should be eliminated.
  // CHECK-NOT: stream.timepoint.await
  %awaited0 = stream.timepoint.await %tp0 => %loaded0
      : !stream.resource<constant>{%c512}
  %awaited1 = stream.timepoint.await %tp1 => %loaded1
      : !stream.resource<constant>{%c1024}

  // Consumer should absorb both timepoints (joined).
  // CHECK: stream.test.timeline_op await(%[[TP0]], %[[TP1]]) => with(%[[LOADED0]], %[[LOADED1]])
  %result, %result_tp = stream.test.timeline_op with(%awaited0, %awaited1)
      : (!stream.resource<constant>{%c512}, !stream.resource<constant>{%c1024})
      -> !stream.resource<external>{%c1024}
      => !stream.timepoint
  %result_awaited = stream.timepoint.await %result_tp => %result
      : !stream.resource<external>{%c1024}
  util.return %result_awaited : !stream.resource<external>
}

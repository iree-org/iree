// RUN: iree-opt --split-input-file --iree-stream-encode-host-tensors %s | FileCheck %s

// CHECK-LABEL: @tensorParameterLoad
// CHECK-SAME: (%[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64)
util.func public @tensorParameterLoad(%key: !util.buffer, %offset: i64) -> !stream.resource<*> {
  %c128 = arith.constant 128 : index
  // CHECK-DAG: %[[SIZE:.+]] = arith.constant 128 : index
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.load %[[KEY]][%[[OFFSET]]] : !stream.resource<*>{%[[SIZE]]} => !stream.timepoint
  // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<*>{%[[SIZE]]}
  %result = stream.tensor.parameter.load %key[%offset] : tensor<4x8xf32> in !stream.resource<*>{%c128}
  // CHECK: util.return %[[AWAITED]]
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @tensorParameterLoadWithScope
// CHECK-SAME: (%[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64)
util.func public @tensorParameterLoadWithScope(%scope: !util.buffer, %key: !util.buffer, %offset: i64) -> !stream.resource<*> {
  %c128 = arith.constant 128 : index
  // CHECK-DAG: %[[SIZE:.+]] = arith.constant 128 : index
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.load %[[SCOPE]]::%[[KEY]][%[[OFFSET]]] : !stream.resource<*>{%[[SIZE]]} => !stream.timepoint
  // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<*>{%[[SIZE]]}
  %result = stream.tensor.parameter.load %scope::%key[%offset] : tensor<4x8xf32> in !stream.resource<*>{%c128}
  // CHECK: util.return %[[AWAITED]]
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @tensorParameterLoadDynamic
// CHECK-SAME: (%[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64, %[[DIM:.+]]: index, %{{.+}}: index)
util.func public @tensorParameterLoadDynamic(%key: !util.buffer, %offset: i64, %dim: index, %size: index) -> !stream.resource<*> {
  // CHECK-DAG: %[[STATIC_SIZE:.+]] = arith.constant 32 : index
  // CHECK: %[[DYNAMIC_SIZE:.+]] = arith.muli %[[DIM]], %[[STATIC_SIZE]] : index
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.load %[[KEY]][%[[OFFSET]]] : !stream.resource<*>{%[[DYNAMIC_SIZE]]} => !stream.timepoint
  // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<*>{%[[DYNAMIC_SIZE]]}
  %result = stream.tensor.parameter.load %key[%offset] : tensor<?x8xf32>{%dim} in !stream.resource<*>{%size}
  // CHECK: util.return %[[AWAITED]]
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @tensorParameterWrite
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<*>, %[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64)
util.func public @tensorParameterWrite(%source: !stream.resource<*>, %key: !util.buffer, %offset: i64) -> !stream.resource<*> {
  %c128 = arith.constant 128 : index
  // CHECK-DAG: %[[ZERO:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[SIZE:.+]] = arith.constant 128 : index
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.write %[[SOURCE]][%[[ZERO]] to %[[SIZE]] for %[[SIZE]]] -> %[[KEY]][%[[OFFSET]]] : !stream.resource<*>{%[[SIZE]]} => !stream.timepoint
  // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<*>{%[[SIZE]]}
  %result = stream.tensor.parameter.write %source -> %key[%offset] : tensor<4x8xf32> in !stream.resource<*>{%c128}
  // CHECK: util.return %[[AWAITED]]
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @tensorParameterWriteWithScope
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<*>, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64)
util.func public @tensorParameterWriteWithScope(%source: !stream.resource<*>, %scope: !util.buffer, %key: !util.buffer, %offset: i64) -> !stream.resource<*> {
  %c128 = arith.constant 128 : index
  // CHECK-DAG: %[[ZERO:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[SIZE:.+]] = arith.constant 128 : index
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.write %[[SOURCE]][%[[ZERO]] to %[[SIZE]] for %[[SIZE]]] -> %[[SCOPE]]::%[[KEY]][%[[OFFSET]]] : !stream.resource<*>{%[[SIZE]]} => !stream.timepoint
  // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<*>{%[[SIZE]]}
  %result = stream.tensor.parameter.write %source -> %scope::%key[%offset] : tensor<4x8xf32> in !stream.resource<*>{%c128}
  // CHECK: util.return %[[AWAITED]]
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @tensorParameterWriteDynamic
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index, %[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64, %{{.+}}: index)
util.func public @tensorParameterWriteDynamic(%source: !stream.resource<*>, %size: index, %key: !util.buffer, %offset: i64, %dim: index) -> !stream.resource<*> {
  // CHECK-DAG: %[[ZERO:.+]] = arith.constant 0 : index
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.write %[[SOURCE]][%[[ZERO]] to %[[SIZE]] for %[[SIZE]]] -> %[[KEY]][%[[OFFSET]]] : !stream.resource<*>{%[[SIZE]]} => !stream.timepoint
  // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<*>{%[[SIZE]]}
  %result = stream.tensor.parameter.write %source -> %key[%offset] : tensor<?x8xf32>{%dim} in !stream.resource<*>{%size}
  // CHECK: util.return %[[AWAITED]]
  util.return %result : !stream.resource<*>
}

// -----

// Verifies that sub-byte element types are properly aligned to byte boundaries
// during encoding. tensor<32xi4> should produce a 16-byte resource (32 * 4 bits
// = 128 bits = 16 bytes), not the 32 bytes that would result from treating i4
// as a full byte.

// CHECK-LABEL: @tensorParameterLoadI4
// CHECK-SAME: (%[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64)
util.func public @tensorParameterLoadI4(%key: !util.buffer, %offset: i64) -> !stream.resource<*> {
  %c32 = arith.constant 32 : index
  // CHECK-DAG: %[[SIZE:.+]] = arith.constant 16 : index
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.load %[[KEY]][%[[OFFSET]]] : !stream.resource<*>{%[[SIZE]]} => !stream.timepoint
  // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<*>{%[[SIZE]]}
  %result = stream.tensor.parameter.load %key[%offset] : tensor<32xi4> in !stream.resource<*>{%c32}
  // CHECK: util.return %[[AWAITED]]
  util.return %result : !stream.resource<*>
}

// -----

// Verifies i1 alignment: tensor<16xi1> produces 16 bytes because i1 elements
// are promoted to i8 (i1 is not sub-byte packed without an explicit packed
// storage attribute), so each element occupies a full byte.

// CHECK-LABEL: @tensorParameterLoadI1
// CHECK-SAME: (%[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64)
util.func public @tensorParameterLoadI1(%key: !util.buffer, %offset: i64) -> !stream.resource<*> {
  %c16 = arith.constant 16 : index
  // CHECK-DAG: %[[SIZE:.+]] = arith.constant 16 : index
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.load %[[KEY]][%[[OFFSET]]] : !stream.resource<*>{%[[SIZE]]} => !stream.timepoint
  %result = stream.tensor.parameter.load %key[%offset] : tensor<16xi1> in !stream.resource<*>{%c16}
  util.return %result : !stream.resource<*>
}

// -----

// Verifies dynamic dimension with sub-byte type: tensor<?xi4>{%dim} should
// produce ceildivui(dim, 2) bytes. i4 elements are sub-byte packed (2 per
// byte), and ceildivui handles odd element counts correctly.

// CHECK-LABEL: @tensorParameterLoadI4Dynamic
// CHECK-SAME: (%[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64, %[[DIM:.+]]: index, %{{.+}}: index)
util.func public @tensorParameterLoadI4Dynamic(%key: !util.buffer, %offset: i64, %dim: index, %size: index) -> !stream.resource<*> {
  // CHECK-DAG: %[[TWO:.+]] = arith.constant 2 : index
  // CHECK: %[[PACKED:.+]] = arith.ceildivui %[[DIM]], %[[TWO]] : index
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.load %[[KEY]][%[[OFFSET]]] : !stream.resource<*>{%[[PACKED]]} => !stream.timepoint
  %result = stream.tensor.parameter.load %key[%offset] : tensor<?xi4>{%dim} in !stream.resource<*>{%size}
  util.return %result : !stream.resource<*>
}

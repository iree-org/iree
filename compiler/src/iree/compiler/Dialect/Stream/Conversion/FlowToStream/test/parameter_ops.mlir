// RUN: iree-opt --split-input-file --iree-stream-conversion %s | FileCheck %s

// CHECK-LABEL: @parameterLoad
// CHECK-SAME: (%[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64)
util.func public @parameterLoad(%key: !util.buffer, %offset: i64) -> tensor<4x8xf32> {
  // CHECK-DAG: %[[SIZE:.+]] = stream.tensor.sizeof tensor<4x8xf32> : index
  // CHECK: %[[RESULT:.+]] = stream.tensor.parameter.load %[[KEY]][%[[OFFSET]]] : tensor<4x8xf32> in !stream.resource<*>{%[[SIZE]]}
  %result = flow.parameter.load %key[%offset] : tensor<4x8xf32>
  // CHECK: util.return %[[RESULT]], %[[SIZE]]
  util.return %result : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @parameterLoadWithScope
// CHECK-SAME: (%[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64)
util.func public @parameterLoadWithScope(%scope: !util.buffer, %key: !util.buffer, %offset: i64) -> tensor<4x8xf32> {
  // CHECK-DAG: %[[SIZE:.+]] = stream.tensor.sizeof tensor<4x8xf32> : index
  // CHECK: %[[RESULT:.+]] = stream.tensor.parameter.load %[[SCOPE]]::%[[KEY]][%[[OFFSET]]] : tensor<4x8xf32> in !stream.resource<*>{%[[SIZE]]}
  %result = flow.parameter.load %scope::%key[%offset] : tensor<4x8xf32>
  // CHECK: util.return %[[RESULT]], %[[SIZE]]
  util.return %result : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @parameterLoadDynamic
// CHECK-SAME: (%[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64, %[[DIM:.+]]: index)
util.func public @parameterLoadDynamic(%key: !util.buffer, %offset: i64, %dim: index) -> tensor<?x8xf32> {
  // CHECK-DAG: %[[SIZE:.+]] = stream.tensor.sizeof tensor<?x8xf32>{%[[DIM]]} : index
  // CHECK: %[[RESULT:.+]] = stream.tensor.parameter.load %[[KEY]][%[[OFFSET]]] : tensor<?x8xf32>{%[[DIM]]} in !stream.resource<*>{%[[SIZE]]}
  %result = flow.parameter.load %key[%offset] : tensor<?x8xf32>{%dim}
  // CHECK: util.return %[[RESULT]], %[[SIZE]]
  util.return %result : tensor<?x8xf32>
}

// -----

// CHECK-LABEL: @parameterWrite
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<*>, %[[SOURCE_SIZE:.+]]: index, %[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64)
util.func public @parameterWrite(%source: tensor<4x8xf32>, %key: !util.buffer, %offset: i64) -> tensor<4x8xf32> {
  // CHECK: %[[RESULT:.+]] = stream.tensor.parameter.write %[[SOURCE]] -> %[[KEY]][%[[OFFSET]]] : tensor<4x8xf32> in !stream.resource<*>{%[[SOURCE_SIZE]]}
  %result = flow.parameter.write %source -> %key[%offset] : tensor<4x8xf32>
  // CHECK: util.return %[[RESULT]], %[[SOURCE_SIZE]]
  util.return %result : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @parameterWriteWithScope
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<*>, %[[SOURCE_SIZE:.+]]: index, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64)
util.func public @parameterWriteWithScope(%source: tensor<4x8xf32>, %scope: !util.buffer, %key: !util.buffer, %offset: i64) -> tensor<4x8xf32> {
  // CHECK: %[[RESULT:.+]] = stream.tensor.parameter.write %[[SOURCE]] -> %[[SCOPE]]::%[[KEY]][%[[OFFSET]]] : tensor<4x8xf32> in !stream.resource<*>{%[[SOURCE_SIZE]]}
  %result = flow.parameter.write %source -> %scope::%key[%offset] : tensor<4x8xf32>
  // CHECK: util.return %[[RESULT]], %[[SOURCE_SIZE]]
  util.return %result : tensor<4x8xf32>
}

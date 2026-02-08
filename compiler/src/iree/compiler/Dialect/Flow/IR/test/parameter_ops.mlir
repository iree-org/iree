// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @parameterLoad
// CHECK-SAME: (%[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64)
util.func public @parameterLoad(%key: !util.buffer, %offset: i64) -> tensor<4x8xf32> {
  // CHECK: %[[RESULT:.+]] = flow.parameter.load %[[KEY]][%[[OFFSET]]] : tensor<4x8xf32>
  %result = flow.parameter.load %key[%offset] : tensor<4x8xf32>
  util.return %result : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @parameterLoadWithScope
// CHECK-SAME: (%[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64)
util.func public @parameterLoadWithScope(%scope: !util.buffer, %key: !util.buffer, %offset: i64) -> tensor<4x8xf32> {
  // CHECK: %[[RESULT:.+]] = flow.parameter.load %[[SCOPE]]::%[[KEY]][%[[OFFSET]]] : tensor<4x8xf32>
  %result = flow.parameter.load %scope::%key[%offset] : tensor<4x8xf32>
  util.return %result : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @parameterLoadDynamic
// CHECK-SAME: (%[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64, %[[DIM:.+]]: index)
util.func public @parameterLoadDynamic(%key: !util.buffer, %offset: i64, %dim: index) -> tensor<?x8xf32> {
  // CHECK: %[[RESULT:.+]] = flow.parameter.load %[[KEY]][%[[OFFSET]]] : tensor<?x8xf32>{%[[DIM]]}
  %result = flow.parameter.load %key[%offset] : tensor<?x8xf32>{%dim}
  util.return %result : tensor<?x8xf32>
}

// -----

// CHECK-LABEL: @parameterWrite
// CHECK-SAME: (%[[SOURCE:.+]]: tensor<4x8xf32>, %[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64)
util.func public @parameterWrite(%source: tensor<4x8xf32>, %key: !util.buffer, %offset: i64) -> tensor<4x8xf32> {
  // CHECK: %[[RESULT:.+]] = flow.parameter.write %[[SOURCE]] -> %[[KEY]][%[[OFFSET]]] : tensor<4x8xf32>
  %result = flow.parameter.write %source -> %key[%offset] : tensor<4x8xf32>
  util.return %result : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @parameterWriteWithScope
// CHECK-SAME: (%[[SOURCE:.+]]: tensor<4x8xf32>, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64)
util.func public @parameterWriteWithScope(%source: tensor<4x8xf32>, %scope: !util.buffer, %key: !util.buffer, %offset: i64) -> tensor<4x8xf32> {
  // CHECK: %[[RESULT:.+]] = flow.parameter.write %[[SOURCE]] -> %[[SCOPE]]::%[[KEY]][%[[OFFSET]]] : tensor<4x8xf32>
  %result = flow.parameter.write %source -> %scope::%key[%offset] : tensor<4x8xf32>
  util.return %result : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: @parameterWriteDynamic
// CHECK-SAME: (%[[SOURCE:.+]]: tensor<?x8xf32>, %[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64, %[[DIM:.+]]: index)
util.func public @parameterWriteDynamic(%source: tensor<?x8xf32>, %key: !util.buffer, %offset: i64, %dim: index) -> tensor<?x8xf32> {
  // CHECK: %[[RESULT:.+]] = flow.parameter.write %[[SOURCE]] -> %[[KEY]][%[[OFFSET]]] : tensor<?x8xf32>{%[[DIM]]}
  %result = flow.parameter.write %source -> %key[%offset] : tensor<?x8xf32>{%dim}
  util.return %result : tensor<?x8xf32>
}

// RUN: iree-opt --split-input-file %s --verify-diagnostics | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @tensorParameterLoad
// CHECK-SAME: (%[[SIZE:.+]]: index, %[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64)
util.func private @tensorParameterLoad(%size: index, %key: !util.buffer, %offset: i64) -> !stream.resource<*> {
  // CHECK: %[[RESULT:.+]] = stream.tensor.parameter.load %[[KEY]][%[[OFFSET]]] : tensor<4x8xf32> in !stream.resource<*>{%[[SIZE]]}
  %result = stream.tensor.parameter.load %key[%offset] : tensor<4x8xf32> in !stream.resource<*>{%size}
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @tensorParameterLoadWithScope
// CHECK-SAME: (%[[SIZE:.+]]: index, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64)
util.func private @tensorParameterLoadWithScope(%size: index, %scope: !util.buffer, %key: !util.buffer, %offset: i64) -> !stream.resource<*> {
  // CHECK: %[[RESULT:.+]] = stream.tensor.parameter.load %[[SCOPE]]::%[[KEY]][%[[OFFSET]]] : tensor<4x8xf32> in !stream.resource<*>{%[[SIZE]]}
  %result = stream.tensor.parameter.load %scope::%key[%offset] : tensor<4x8xf32> in !stream.resource<*>{%size}
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @tensorParameterLoadDynamic
// CHECK-SAME: (%[[SIZE:.+]]: index, %[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64, %[[DIM:.+]]: index)
util.func private @tensorParameterLoadDynamic(%size: index, %key: !util.buffer, %offset: i64, %dim: index) -> !stream.resource<*> {
  // CHECK: %[[RESULT:.+]] = stream.tensor.parameter.load %[[KEY]][%[[OFFSET]]] : tensor<?x8xf32>{%[[DIM]]} in !stream.resource<*>{%[[SIZE]]}
  %result = stream.tensor.parameter.load %key[%offset] : tensor<?x8xf32>{%dim} in !stream.resource<*>{%size}
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @tensorParameterWrite
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index, %[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64)
util.func private @tensorParameterWrite(%source: !stream.resource<*>, %size: index, %key: !util.buffer, %offset: i64) -> !stream.resource<*> {
  // CHECK: %[[RESULT:.+]] = stream.tensor.parameter.write %[[SOURCE]] -> %[[KEY]][%[[OFFSET]]] : tensor<4x8xf32> in !stream.resource<*>{%[[SIZE]]}
  %result = stream.tensor.parameter.write %source -> %key[%offset] : tensor<4x8xf32> in !stream.resource<*>{%size}
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @tensorParameterWriteWithScope
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index, %[[SCOPE:.+]]: !util.buffer, %[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64)
util.func private @tensorParameterWriteWithScope(%source: !stream.resource<*>, %size: index, %scope: !util.buffer, %key: !util.buffer, %offset: i64) -> !stream.resource<*> {
  // CHECK: %[[RESULT:.+]] = stream.tensor.parameter.write %[[SOURCE]] -> %[[SCOPE]]::%[[KEY]][%[[OFFSET]]] : tensor<4x8xf32> in !stream.resource<*>{%[[SIZE]]}
  %result = stream.tensor.parameter.write %source -> %scope::%key[%offset] : tensor<4x8xf32> in !stream.resource<*>{%size}
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @tensorParameterWriteDynamic
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index, %[[KEY:.+]]: !util.buffer, %[[OFFSET:.+]]: i64, %[[DIM:.+]]: index)
util.func private @tensorParameterWriteDynamic(%source: !stream.resource<*>, %size: index, %key: !util.buffer, %offset: i64, %dim: index) -> !stream.resource<*> {
  // CHECK: %[[RESULT:.+]] = stream.tensor.parameter.write %[[SOURCE]] -> %[[KEY]][%[[OFFSET]]] : tensor<?x8xf32>{%[[DIM]]} in !stream.resource<*>{%[[SIZE]]}
  %result = stream.tensor.parameter.write %source -> %key[%offset] : tensor<?x8xf32>{%dim} in !stream.resource<*>{%size}
  util.return %result : !stream.resource<*>
}

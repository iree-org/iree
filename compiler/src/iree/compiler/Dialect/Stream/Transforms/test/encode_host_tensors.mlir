// RUN: iree-opt --split-input-file --iree-stream-encode-host-tensors %s | FileCheck %s

// CHECK-LABEL: @denseTensorSizeOf
util.func public @denseTensorSizeOf(%arg0: index) -> index {
  // CHECK: %[[STATIC_SIZE:.+]] = arith.constant 20 : index
  // CHECK: %[[DYNAMIC_SIZE:.+]] = arith.muli %arg0, %[[STATIC_SIZE]] : index
  %dynamic_size = stream.tensor.sizeof tensor<?x5xf32>{%arg0} : index
  // CHECK: util.return %[[DYNAMIC_SIZE]]
  util.return %dynamic_size : index
}

// -----

// CHECK-LABEL: @denseTensorSizeOfEmpty
util.func public @denseTensorSizeOfEmpty(%arg0: index) -> index {
  // CHECK: %[[ZERO_SIZE:.+]] = arith.constant 0 : index
  %zero_size = stream.tensor.sizeof tensor<?x0xf32>{%arg0} : index
  // CHECK: util.return %[[ZERO_SIZE]]
  util.return %zero_size : index
}

// -----

// CHECK-LABEL: @denseTensorEmpty
util.func public @denseTensorEmpty(%arg0: index, %arg1: index) -> !stream.resource<*> {
  // CHECK: %[[RESULT:.+]] = stream.async.alloca : !stream.resource<*>{%arg1}
  %result = stream.tensor.empty : tensor<?x1xf32>{%arg0} in !stream.resource<*>{%arg1}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorConstant
util.func public @denseTensorConstant(%arg0: index) -> !stream.resource<constant> {
  // CHECK: %[[STATIC_SIZE:.+]] = arith.constant 1280 : index
  // CHECK: %[[DYNAMIC_SIZE:.+]] = arith.muli %arg0, %[[STATIC_SIZE]] : index
  // CHECK: %[[RESULT:.+]] = stream.async.constant : !stream.resource<constant>{%[[DYNAMIC_SIZE]]} = dense<0.000000e+00> : tensor<1x5x64xf32>
  %result = stream.tensor.constant : tensor<?x5x64xf32>{%arg0} in !stream.resource<constant> = dense<0.000000e+00> : tensor<1x5x64xf32>
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<constant>
}

// -----

// Tests that sub-byte element width constants get extended to byte alignment.

// CHECK-LABEL: @denseTensorConstantI1
util.func public @denseTensorConstantI1() -> !stream.resource<constant> {
  // CHECK: %[[STATIC_SIZE:.+]] = arith.constant 4 : index
  // CHECK: %[[RESULT:.+]] = stream.async.constant : !stream.resource<constant>{%[[STATIC_SIZE]]} = dense<[1, 1, 0, 1]> : tensor<4xi8>
  %result = stream.tensor.constant : tensor<4xi1> in !stream.resource<constant> = dense<[true, true, false, true]> : tensor<4xi1>
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @denseTensorSplatI32
util.func public @denseTensorSplatI32(%arg0: i32, %arg1: index, %arg2: index) -> !stream.resource<*> {
  // CHECK: %[[RESULT:.+]] = stream.async.splat %arg0 : i32 -> !stream.resource<*>{%arg2}
  %result = stream.tensor.splat %arg0 : i32 -> tensor<?x1x10xi32>{%arg1} in !stream.resource<*>{%arg2}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorSplatI1
util.func public @denseTensorSplatI1(%arg0: i1, %arg1: index, %arg2: index) -> !stream.resource<*> {
  // CHECK: %[[PATTERN:.+]] = arith.extui %arg0 : i1 to i8
  // CHECK: %[[RESULT:.+]] = stream.async.splat %[[PATTERN]] : i8 -> !stream.resource<*>{%arg2}
  %result = stream.tensor.splat %arg0 : i1 -> tensor<?x1x10xi1>{%arg1} in !stream.resource<*>{%arg2}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorSplatBF16
util.func public @denseTensorSplatBF16(%arg0: bf16, %arg1: index, %arg2: index) -> !stream.resource<*> {
  // CHECK: %[[PATTERN:.+]] = arith.bitcast %arg0 : bf16 to i16
  // CHECK: %[[RESULT:.+]] = stream.async.splat %[[PATTERN]] : i16 -> !stream.resource<*>{%arg2}
  %result = stream.tensor.splat %arg0 : bf16 -> tensor<?x1x10xbf16>{%arg1} in !stream.resource<*>{%arg2}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorSplatF32
util.func public @denseTensorSplatF32(%arg0: f32, %arg1: index, %arg2: index) -> !stream.resource<*> {
  // CHECK: %[[PATTERN:.+]] = arith.bitcast %arg0 : f32 to i32
  // CHECK: %[[RESULT:.+]] = stream.async.splat %[[PATTERN]] : i32 -> !stream.resource<*>{%arg2}
  %result = stream.tensor.splat %arg0 : f32 -> tensor<?x1x10xf32>{%arg1} in !stream.resource<*>{%arg2}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorSplatI64
util.func public @denseTensorSplatI64(%arg0: i64, %arg1: index, %arg2: index) -> !stream.resource<*> {
  // CHECK: %[[RESULT:.+]] = stream.async.splat %arg0 : i64 -> !stream.resource<*>{%arg2}
  %result = stream.tensor.splat %arg0 : i64 -> tensor<?x1x10xi64>{%arg1} in !stream.resource<*>{%arg2}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorSplatConstantComplexF32
util.func public @denseTensorSplatConstantComplexF32(%arg0: !stream.resource<*>) -> (!stream.resource<*>) {
  %cst = complex.constant [3.000000e+00 : f32, 1.000000e+01 : f32] : complex<f32>
  %result_size = stream.tensor.sizeof tensor<6xcomplex<f32>> : index
  // CHECK: %[[I64NUMBER:.+]] = complex.constant [3.000000e+00 : f32, 1.000000e+01 : f32] : complex<f32>
  // CHECK: %[[BITCAST:.+]] = complex.bitcast %[[I64NUMBER]] : complex<f32> to i64
  // CHECK: %[[RESULT:.+]] = stream.async.splat %[[BITCAST]]
  %result = stream.tensor.splat %cst : complex<f32> -> tensor<6xcomplex<f32>> in !stream.resource<*>{%result_size}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorSplatDynamicComplexF32
util.func public @denseTensorSplatDynamicComplexF32(%arg0: !stream.resource<*>, %arg1: complex<f32>) -> (!stream.resource<*>) {
  %result_size = stream.tensor.sizeof tensor<6xcomplex<f32>> : index
  // CHECK: %[[BITCAST:.+]] = complex.bitcast %arg1 : complex<f32> to i64
  // CHECK: %[[RESULT:.+]] = stream.async.splat %[[BITCAST]]
  %result = stream.tensor.splat %arg1 : complex<f32> -> tensor<6xcomplex<f32>> in !stream.resource<*>{%result_size}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}

// -----

// NOTE: clone likes to fold; the fills ensure it doesn't.

// CHECK-LABEL: @denseTensorClone
util.func public @denseTensorClone(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: f32) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[CLONE:.+]] = stream.async.clone %arg0 : !stream.resource<*>{%arg2} -> !stream.resource<*>{%arg2}
  %clone = stream.tensor.clone %arg0 : tensor<?x4xf32>{%arg1} in !stream.resource<*>{%arg2} -> tensor<?x4xf32>{%arg1} in !stream.resource<*>{%arg2}
  // CHECK: %[[FILL:.+]] = stream.async.fill
  %fill = stream.tensor.fill %arg3, %clone[%c0, %c0 for %c1, %c1] : f32 -> tensor<?x4xf32>{%arg1} in %0 as !stream.resource<*>{%arg2}
  // CHECK: util.return %[[CLONE]], %[[FILL]]
  util.return %clone, %fill : !stream.resource<*>, !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorSlice
util.func public @denseTensorSlice(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 4 : index
  // CHECK: %[[END:.+]] = arith.addi %arg4, %[[OFFSET]] : index
  // CHECK: %[[RESULT:.+]] = stream.async.slice %arg0[%[[OFFSET]] to %[[END]]] : !stream.resource<*>{%arg2} -> !stream.resource<*>{%arg4}
  %result = stream.tensor.slice %arg0[%c0, %c1 for %arg3, %c1] : tensor<?x4xf32>{%arg1} in !stream.resource<*>{%arg2} -> tensor<?x1xf32>{%arg3} in !stream.resource<*>{%arg4}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorFillF32
util.func public @denseTensorFillF32(%arg0: f32, %arg1: !stream.resource<*>, %arg2: index, %arg3: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-DAG: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 20 : index
  // CHECK-DAG: %[[PATTERN:.+]] = arith.bitcast %arg0 : f32 to i32
  // CHECK: %[[RESULT:.+]] = stream.async.fill %[[PATTERN]], %arg1[%[[OFFSET]] to %[[LENGTH]] for %[[LENGTH]]] : i32 -> %arg1 as !stream.resource<*>{%arg3}
  %result = stream.tensor.fill %arg0, %arg1[%c0, %c0 for %c1, %c1] : f32 -> tensor<?x4xf32>{%arg2} in %arg1 as !stream.resource<*>{%arg3}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorFillI64
util.func public @denseTensorFillI64(%arg0: i64, %arg1: !stream.resource<*>, %arg2: index, %arg3: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-DAG: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 40 : index
  // CHECK: %[[RESULT:.+]] = stream.async.fill %arg0, %arg1[%[[OFFSET]] to %[[LENGTH]] for %[[LENGTH]]] : i64 -> %arg1 as !stream.resource<*>{%arg3}
  %result = stream.tensor.fill %arg0, %arg1[%c0, %c0 for %c1, %c1] : i64 -> tensor<?x4xi64>{%arg2} in %arg1 as !stream.resource<*>{%arg3}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorFillF64
util.func public @denseTensorFillF64(%arg0: f64, %arg1: !stream.resource<*>, %arg2: index, %arg3: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-DAG: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 40 : index
  // CHECK-DAG: %[[PATTERN:.+]] = arith.bitcast %arg0 : f64 to i64
  // CHECK: %[[RESULT:.+]] = stream.async.fill %[[PATTERN]], %arg1[%[[OFFSET]] to %[[LENGTH]] for %[[LENGTH]]] : i64 -> %arg1 as !stream.resource<*>{%arg3}
  %result = stream.tensor.fill %arg0, %arg1[%c0, %c0 for %c1, %c1] : f64 -> tensor<?x4xi64>{%arg2} in %arg1 as !stream.resource<*>{%arg3}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorUpdate
util.func public @denseTensorUpdate(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.resource<*>, %arg3: index, %arg4: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[RESULT:.+]] = stream.async.update %arg0, %arg2[%[[OFFSET]] to %arg1] : !stream.resource<*>{%arg1} -> %arg2 as !stream.resource<*>{%arg4}
  %result = stream.tensor.update %arg0, %arg2[%c0, %c0] : tensor<2x2xf32> in !stream.resource<*>{%arg1} -> tensor<?x4xf32>{%arg3} in %arg2 as !stream.resource<*>{%arg4}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorLoad
util.func public @denseTensorLoad(%arg0: !stream.resource<staging>, %arg1: index, %arg2: index) -> f32 {
  %c0 = arith.constant 0 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[RESULT:.+]] = stream.async.load %arg0[%[[OFFSET]]] : !stream.resource<staging>{%arg2} -> f32
  %result = stream.tensor.load %arg0[%c0] : tensor<?xf32>{%arg1} in !stream.resource<staging>{%arg2} -> f32
  // CHECK: util.return %[[RESULT]]
  util.return %result : f32
}

// -----

// CHECK-LABEL: @denseTensorLoadRank0
util.func public @denseTensorLoadRank0(%arg0: !stream.resource<staging>, %arg1: index) -> f32 {
  %c0 = arith.constant 0 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[RESULT:.+]] = stream.async.load %arg0[%[[OFFSET]]] : !stream.resource<staging>{%arg1} -> f32
  %result = stream.tensor.load %arg0 : tensor<f32> in !stream.resource<staging>{%arg1} -> f32
  // CHECK: util.return %[[RESULT]]
  util.return %result : f32
}

// -----

// CHECK-LABEL: @denseTensorStore
util.func public @denseTensorStore(%arg0: !stream.resource<staging>, %arg1: index, %arg2: index, %arg3: f32) -> !stream.resource<staging> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[RESULT:.+]] = stream.async.store %arg3, %arg0[%[[OFFSET]]] : f32 -> %arg0 as !stream.resource<staging>{%arg2}
  %result = stream.tensor.store %arg3, %arg0[%c0] : f32 -> tensor<?xf32>{%arg1} in %arg0 as !stream.resource<staging>{%arg2}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<staging>
}

// -----

// CHECK-LABEL: @denseTensorStoreRank0
util.func public @denseTensorStoreRank0(%arg0: !stream.resource<staging>, %arg1: index, %arg2: f32) -> !stream.resource<staging> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[RESULT:.+]] = stream.async.store %arg2, %arg0[%[[OFFSET]]] : f32 -> %arg0 as !stream.resource<staging>{%arg1}
  %result = stream.tensor.store %arg2, %arg0 : f32 -> tensor<f32> in %arg0 as !stream.resource<staging>{%arg1}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<staging>
}

// -----

// CHECK-LABEL: @denseTensorDispatch
// CHECK-SAME: (%[[RESOURCE0:.+]]: !stream.resource<transient>, %[[RESOURCE0_SIZE:[a-z0-9]+]]: index, %[[TENSOR0_DIM:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[RESOURCE1:.+]]: !stream.resource<external>, %[[RESOURCE1_SIZE:[a-z0-9]+]]: index, %[[TENSOR1_DIM:[a-z0-9]+]]: index)
util.func public @denseTensorDispatch(
    %resource0: !stream.resource<transient>, %resource0_size: index, %tensor0_dim: index,
    %resource1: !stream.resource<external>, %resource1_size: index, %tensor1_dim: index) -> (!stream.resource<external>, !stream.resource<external>) {
  // CHECK: %[[ZERO:.+]] = arith.constant 0
  // CHECK: %[[RESULTS:.+]]:2 = stream.async.dispatch @ex::@entry
  // CHECK-SAME: (%[[RESOURCE0]][%[[ZERO]] to %[[RESOURCE0_SIZE]] for %[[RESOURCE0_SIZE]]],
  // CHECK-SAME:  %[[RESOURCE1]][%[[ZERO]] to %[[RESOURCE1_SIZE]] for %[[RESOURCE1_SIZE]]])
  // CHECK-SAME: (!stream.resource<transient>{%[[RESOURCE0_SIZE]]}, !stream.resource<external>{%[[RESOURCE1_SIZE]]}) ->
  // CHECK-SAME: (!stream.resource<external>{%[[RESOURCE1_SIZE]]}, %[[RESOURCE1]]{%[[RESOURCE1_SIZE]]})
  %results:2 = stream.tensor.dispatch @ex::@entry(%resource0, %resource1) : (tensor<4x?xf32>{%tensor0_dim} in !stream.resource<transient>{%resource0_size}, tensor<?xi32>{%tensor1_dim} in !stream.resource<external>{%resource1_size}) -> (tensor<4x?xf32>{%tensor0_dim} in !stream.resource<external>{%resource1_size}, tensor<?xi32>{%tensor1_dim} in %resource1{%resource1_size})
  // CHECK: util.return %[[RESULTS]]#0, %[[RESULTS]]#1
  util.return %results#0, %results#1 : !stream.resource<external>, !stream.resource<external>
}

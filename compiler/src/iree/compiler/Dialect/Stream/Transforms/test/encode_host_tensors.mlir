// RUN: iree-opt --split-input-file --iree-stream-encode-host-tensors %s | FileCheck %s

// CHECK-LABEL: @denseTensorSizeOf
util.func public @denseTensorSizeOf(%arg0: index) -> index {
  // CHECK: %[[STATIC_SIZE:.+]] = arith.constant 20 : index
  // CHECK: %[[DYNAMIC_SIZE:.+]] = arith.muli %arg0, %[[STATIC_SIZE]] : index
  %0 = stream.tensor.sizeof tensor<?x5xf32>{%arg0} : index
  // CHECK: util.return %[[DYNAMIC_SIZE]]
  util.return %0 : index
}

// -----

// CHECK-LABEL: @denseTensorSizeOfEmpty
util.func public @denseTensorSizeOfEmpty(%arg0: index) -> index {
  // CHECK: %[[ZERO:.+]] = arith.constant 0 : index
  %0 = stream.tensor.sizeof tensor<?x0xf32>{%arg0} : index
  // CHECK: util.return %[[ZERO]]
  util.return %0 : index
}

// -----

// CHECK-LABEL: @denseTensorEmpty
util.func public @denseTensorEmpty(%arg0: index, %arg1: index) -> !stream.resource<*> {
  // CHECK: %[[RET:.+]] = stream.async.alloca : !stream.resource<*>{%arg1}
  %0 = stream.tensor.empty : tensor<?x1xf32>{%arg0} in !stream.resource<*>{%arg1}
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorConstant
util.func public @denseTensorConstant(%arg0: index) -> !stream.resource<constant> {
  // CHECK: %[[STATIC_SIZE:.+]] = arith.constant 1280 : index
  // CHECK: %[[DYNAMIC_SIZE:.+]] = arith.muli %arg0, %[[STATIC_SIZE]] : index
  // CHECK: %[[RET:.+]] = stream.async.constant : !stream.resource<constant>{%[[DYNAMIC_SIZE]]} = dense<0.000000e+00> : tensor<1x5x64xf32>
  %0 = stream.tensor.constant : tensor<?x5x64xf32>{%arg0} in !stream.resource<constant> = dense<0.000000e+00> : tensor<1x5x64xf32>
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<constant>
}

// -----

// Tests that sub-byte element width constants get extended to byte alignment.

// CHECK-LABEL: @denseTensorConstantI1
util.func public @denseTensorConstantI1() -> !stream.resource<constant> {
  // CHECK: %[[STATIC_SIZE:.+]] = arith.constant 4 : index
  // CHECK: %[[RET:.+]] = stream.async.constant : !stream.resource<constant>{%[[STATIC_SIZE]]} = dense<[1, 1, 0, 1]> : tensor<4xi8>
  %0 = stream.tensor.constant : tensor<4xi1> in !stream.resource<constant> = dense<[true, true, false, true]> : tensor<4xi1>
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @denseTensorSplatI32
util.func public @denseTensorSplatI32(%arg0: i32, %arg1: index, %arg2: index) -> !stream.resource<*> {
  // CHECK: %[[RET:.+]] = stream.async.splat %arg0 : i32 -> !stream.resource<*>{%arg2}
  %0 = stream.tensor.splat %arg0 : i32 -> tensor<?x1x10xi32>{%arg1} in !stream.resource<*>{%arg2}
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorSplatI1
util.func public @denseTensorSplatI1(%arg0: i1, %arg1: index, %arg2: index) -> !stream.resource<*> {
  // CHECK: %[[PATTERN:.+]] = arith.extui %arg0 : i1 to i8
  // CHECK: %[[RET:.+]] = stream.async.splat %[[PATTERN]] : i8 -> !stream.resource<*>{%arg2}
  %0 = stream.tensor.splat %arg0 : i1 -> tensor<?x1x10xi1>{%arg1} in !stream.resource<*>{%arg2}
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorSplatBF16
util.func public @denseTensorSplatBF16(%arg0: bf16, %arg1: index, %arg2: index) -> !stream.resource<*> {
  // CHECK: %[[PATTERN:.+]] = arith.bitcast %arg0 : bf16 to i16
  // CHECK: %[[RET:.+]] = stream.async.splat %[[PATTERN]] : i16 -> !stream.resource<*>{%arg2}
  %0 = stream.tensor.splat %arg0 : bf16 -> tensor<?x1x10xbf16>{%arg1} in !stream.resource<*>{%arg2}
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorSplatF32
util.func public @denseTensorSplatF32(%arg0: f32, %arg1: index, %arg2: index) -> !stream.resource<*> {
  // CHECK: %[[PATTERN:.+]] = arith.bitcast %arg0 : f32 to i32
  // CHECK: %[[RET:.+]] = stream.async.splat %[[PATTERN]] : i32 -> !stream.resource<*>{%arg2}
  %0 = stream.tensor.splat %arg0 : f32 -> tensor<?x1x10xf32>{%arg1} in !stream.resource<*>{%arg2}
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorSplatI64
util.func public @denseTensorSplatI64(%arg0: i64, %arg1: index, %arg2: index) -> !stream.resource<*> {
  // CHECK: %[[RET:.+]] = stream.async.splat %arg0 : i64 -> !stream.resource<*>{%arg2}
  %0 = stream.tensor.splat %arg0 : i64 -> tensor<?x1x10xi64>{%arg1} in !stream.resource<*>{%arg2}
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorSplatConstantComplexF32
util.func public @denseTensorSplatConstantComplexF32(%arg0: !stream.resource<*>) -> (!stream.resource<*>) {
  %cst = complex.constant [3.000000e+00 : f32, 1.000000e+01 : f32] : complex<f32>
  %0 = stream.tensor.sizeof tensor<6xcomplex<f32>> : index
  // CHECK: %[[I64NUMBER:.+]] = complex.constant [3.000000e+00 : f32, 1.000000e+01 : f32] : complex<f32>
  // CHECK: %[[BITCAST:.+]] = complex.bitcast %[[I64NUMBER]] : complex<f32> to i64
  // CHECK: %[[SPLAT_RES:.+]] = stream.async.splat %[[BITCAST]]
  %1 = stream.tensor.splat %cst : complex<f32> -> tensor<6xcomplex<f32>> in !stream.resource<*>{%0}
  // CHECK: util.return %[[SPLAT_RES]]
  util.return %1 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorSplatDynamicComplexF32
util.func public @denseTensorSplatDynamicComplexF32(%arg0: !stream.resource<*>, %arg1: complex<f32>) -> (!stream.resource<*>) {
  %0 = stream.tensor.sizeof tensor<6xcomplex<f32>> : index
  // CHECK: %[[BITCAST:.+]] = complex.bitcast %arg1 : complex<f32> to i64
  // CHECK: %[[SPLAT_RES:.+]] = stream.async.splat %[[BITCAST]]
  %1 = stream.tensor.splat %arg1 : complex<f32> -> tensor<6xcomplex<f32>> in !stream.resource<*>{%0}
  // CHECK: util.return %[[SPLAT_RES]]
  util.return %1 : !stream.resource<*>
}

// -----

// NOTE: clone likes to fold; the fills ensure it doesn't.

// CHECK-LABEL: @denseTensorClone
util.func public @denseTensorClone(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: f32) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[RET:.+]] = stream.async.clone %arg0 : !stream.resource<*>{%arg2} -> !stream.resource<*>{%arg2}
  %0 = stream.tensor.clone %arg0 : tensor<?x4xf32>{%arg1} in !stream.resource<*>{%arg2} -> tensor<?x4xf32>{%arg1} in !stream.resource<*>{%arg2}
  %1 = stream.tensor.fill %arg3, %0[%c0, %c0 for %c1, %c1] : f32 -> tensor<?x4xf32>{%arg1} in %0 as !stream.resource<*>{%arg2}
  util.return %0, %1 : !stream.resource<*>, !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorSlice
util.func public @denseTensorSlice(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 4 : index
  // CHECK: %[[END:.+]] = arith.addi %arg4, %[[OFFSET]] : index
  // CHECK: %[[RET:.+]] = stream.async.slice %arg0[%[[OFFSET]] to %[[END]]] : !stream.resource<*>{%arg2} -> !stream.resource<*>{%arg4}
  %0 = stream.tensor.slice %arg0[%c0, %c1 for %arg3, %c1] : tensor<?x4xf32>{%arg1} in !stream.resource<*>{%arg2} -> tensor<?x1xf32>{%arg3} in !stream.resource<*>{%arg4}
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorFillF32
util.func public @denseTensorFillF32(%arg0: f32, %arg1: !stream.resource<*>, %arg2: index, %arg3: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-DAG: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 20 : index
  // CHECK-DAG: %[[PATTERN:.+]] = arith.bitcast %arg0 : f32 to i32
  // CHECK: %[[RET:.+]] = stream.async.fill %[[PATTERN]], %arg1[%[[OFFSET]] to %[[LENGTH]] for %[[LENGTH]]] : i32 -> %arg1 as !stream.resource<*>{%arg3}
  %0 = stream.tensor.fill %arg0, %arg1[%c0, %c0 for %c1, %c1] : f32 -> tensor<?x4xf32>{%arg2} in %arg1 as !stream.resource<*>{%arg3}
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorFillI64
util.func public @denseTensorFillI64(%arg0: i64, %arg1: !stream.resource<*>, %arg2: index, %arg3: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-DAG: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 40 : index
  // CHECK: %[[RET:.+]] = stream.async.fill %arg0, %arg1[%[[OFFSET]] to %[[LENGTH]] for %[[LENGTH]]] : i64 -> %arg1 as !stream.resource<*>{%arg3}
  %0 = stream.tensor.fill %arg0, %arg1[%c0, %c0 for %c1, %c1] : i64 -> tensor<?x4xi64>{%arg2} in %arg1 as !stream.resource<*>{%arg3}
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorFillF64
util.func public @denseTensorFillF64(%arg0: f64, %arg1: !stream.resource<*>, %arg2: index, %arg3: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-DAG: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 40 : index
  // CHECK-DAG: %[[PATTERN:.+]] = arith.bitcast %arg0 : f64 to i64
  // CHECK: %[[RET:.+]] = stream.async.fill %[[PATTERN]], %arg1[%[[OFFSET]] to %[[LENGTH]] for %[[LENGTH]]] : i64 -> %arg1 as !stream.resource<*>{%arg3}
  %0 = stream.tensor.fill %arg0, %arg1[%c0, %c0 for %c1, %c1] : f64 -> tensor<?x4xi64>{%arg2} in %arg1 as !stream.resource<*>{%arg3}
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorUpdate
util.func public @denseTensorUpdate(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.resource<*>, %arg3: index, %arg4: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[RET:.+]] = stream.async.update %arg0, %arg2[%[[OFFSET]] to %arg1] : !stream.resource<*>{%arg1} -> %arg2 as !stream.resource<*>{%arg4}
  %0 = stream.tensor.update %arg0, %arg2[%c0, %c0] : tensor<2x2xf32> in !stream.resource<*>{%arg1} -> tensor<?x4xf32>{%arg3} in %arg2 as !stream.resource<*>{%arg4}
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorLoad
util.func public @denseTensorLoad(%arg0: !stream.resource<staging>, %arg1: index, %arg2: index) -> f32 {
  %c0 = arith.constant 0 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[RET:.+]] = stream.async.load %arg0[%[[OFFSET]]] : !stream.resource<staging>{%arg2} -> f32
  %0 = stream.tensor.load %arg0[%c0] : tensor<?xf32>{%arg1} in !stream.resource<staging>{%arg2} -> f32
  // CHECK: util.return %[[RET]]
  util.return %0 : f32
}

// -----

// CHECK-LABEL: @denseTensorLoadRank0
util.func public @denseTensorLoadRank0(%arg0: !stream.resource<staging>, %arg1: index) -> f32 {
  %c0 = arith.constant 0 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[RET:.+]] = stream.async.load %arg0[%[[OFFSET]]] : !stream.resource<staging>{%arg1} -> f32
  %0 = stream.tensor.load %arg0 : tensor<f32> in !stream.resource<staging>{%arg1} -> f32
  // CHECK: util.return %[[RET]]
  util.return %0 : f32
}

// -----

// CHECK-LABEL: @denseTensorStore
util.func public @denseTensorStore(%arg0: !stream.resource<staging>, %arg1: index, %arg2: index, %arg3: f32) -> !stream.resource<staging> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[RET:.+]] = stream.async.store %arg3, %arg0[%[[OFFSET]]] : f32 -> %arg0 as !stream.resource<staging>{%arg2}
  %0 = stream.tensor.store %arg3, %arg0[%c0] : f32 -> tensor<?xf32>{%arg1} in %arg0 as !stream.resource<staging>{%arg2}
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<staging>
}

// -----

// CHECK-LABEL: @denseTensorStoreRank0
util.func public @denseTensorStoreRank0(%arg0: !stream.resource<staging>, %arg1: index, %arg2: f32) -> !stream.resource<staging> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[RET:.+]] = stream.async.store %arg2, %arg0[%[[OFFSET]]] : f32 -> %arg0 as !stream.resource<staging>{%arg1}
  %0 = stream.tensor.store %arg2, %arg0 : f32 -> tensor<f32> in %arg0 as !stream.resource<staging>{%arg1}
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<staging>
}

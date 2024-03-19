// RUN: iree-opt --split-input-file --iree-stream-encode-host-tensors --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL:  util.func public @denseTensorConstantI2()
util.func public @denseTensorConstantI2() -> !stream.resource<constant> {
  // CHECK: %[[STATIC_SIZE:.+]] = arith.constant 4 : index
  // CHECK: %[[RET:.+]] = stream.async.constant : !stream.resource<constant>{%[[STATIC_SIZE]]} =
  // CHECK-SAME: dense<[0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1]> : tensor<16xi2>
  %0 = stream.tensor.constant : tensor<16xi2> in !stream.resource<constant> = dense<[
    0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
  ]> : tensor<16xi2>
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<constant>
}

// -----

// Ensures that a non-power-of-two type (i3) constant is expanded to a full byte
// because we don't currently do unaligned sub-byte packing.

// CHECK:  util.func public @denseTensorConstantI3()
util.func public @denseTensorConstantI3() -> !stream.resource<constant> {
  // CHECK: %[[STATIC_SIZE:.+]] = arith.constant 4 : index
  // CHECK: %[[RET:.+]] = stream.async.constant : !stream.resource<constant>{%[[STATIC_SIZE]]} = dense<[0, 7, 2, 5]> : tensor<4xi8>
  %0 = stream.tensor.constant : tensor<4xi3> in !stream.resource<constant> = dense<[0, 7, 2, 5]> : tensor<4xi3>
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @denseTensorConstantI4
util.func public @denseTensorConstantI4() -> !stream.resource<constant> {
  // CHECK: %[[STATIC_SIZE:.+]] = arith.constant 4 : index
  // CHECK: %[[RET:.+]] = stream.async.constant : !stream.resource<constant>{%[[STATIC_SIZE]]} = dense<[5, -1, 0, 3, 1, 7, -8, 4]> : tensor<8xi4>
  %0 = stream.tensor.constant : tensor<8xi4> in !stream.resource<constant> = dense<[5, 15, 0, 3, 1, 7, 8, 4]> : tensor<8xi4>
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<constant>
}

// -----

// Checks that non-byte-aligned total size is not supported for constant.

util.func public @denseTensorConstantI4() -> !stream.resource<constant> {
  // expected-error @+1 {{failed to calculate total byte count: 'tensor<5xi4>' does not have integral number of total bytes}}
  %0 = stream.tensor.constant : tensor<5xi4> in !stream.resource<constant> = dense<[5, 15, 0, 3, 1]> : tensor<5xi4>
  util.return %0 : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @denseTensorConstantI8
util.func public @denseTensorConstantI8() -> !stream.resource<constant> {
  // CHECK: %[[STATIC_SIZE:.+]] = arith.constant 8 : index
  // CHECK: %[[RET:.+]] = stream.async.constant : !stream.resource<constant>{%[[STATIC_SIZE]]} = dense<[5, 15, 0, 3, 1, 7, 8, 4]> : tensor<8xi8>
  %0 = stream.tensor.constant : tensor<8xi8> in !stream.resource<constant> = dense<[5, 15, 0, 3, 1, 7, 8, 4]> : tensor<8xi8>
  // CHECK: util.return %[[RET]]
  util.return %0 : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @denseTensorSizeOfStatic
util.func public @denseTensorSizeOfStatic() -> index {
  // CHECK-DAG: %[[C6:.+]] = arith.constant 6 : index
  %0 = stream.tensor.sizeof tensor<12xi4> : index
  // CHECK: util.return %[[C6]]
  util.return %0 : index
}

// -----

// Checks that non-byte-aligned total size is not supported for sizeof.

util.func public @denseTensorSizeOfStatic() -> index {
  // expected-error @+1 {{failed to calculate total byte count: 'tensor<11xi4>' does not have integral number of total bytes}}
  %0 = stream.tensor.sizeof tensor<11xi4> : index
  util.return %0 : index
}

// -----

// CHECK-LABEL: @denseTensorSizeOfDynamic
util.func public @denseTensorSizeOfDynamic(%arg0: index) -> index {
  // CHECK-DAG: %[[C5:.+]] = arith.constant 5 : index
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  // CHECK: %[[MUL:.+]] = arith.muli %arg0, %[[C5]] : index
  // CHECK: %[[DIV:.+]] = arith.divui %[[MUL]], %[[C2]] : index
  %0 = stream.tensor.sizeof tensor<?x5xi4>{%arg0} : index
  // CHECK: util.return %[[DIV]]
  util.return %0 : index
}

// -----

// Checks that stream.tensor.load with sub-byte packing is not supported right now.

// CHECK-LABEL: @denseTensorLoad
util.func public @denseTensorLoad(%arg0: !stream.resource<staging>, %arg1: index, %arg2: index, %arg3: index) -> i4 {
  %c0 = arith.constant 0 : index
  // CHECK: stream.tensor.load
  %0 = stream.tensor.load %arg0[%arg3] : tensor<?xi4>{%arg1} in !stream.resource<staging>{%arg2} -> i4
  util.return %0 : i4
}

// -----

// Checks that stream.tensor.store with sub-byte packing is not supported right now.

// CHECK-LABEL: @denseTensorStore
util.func public @denseTensorStore(%arg0: !stream.resource<staging>, %arg1: index, %arg2: index, %arg3: i4) -> !stream.resource<staging> {
  %c0 = arith.constant 0 : index
  // CHECK: stream.tensor.store
  %0 = stream.tensor.store %arg3, %arg0[%c0] : i4 -> tensor<?xi4>{%arg1} in %arg0 as !stream.resource<staging>{%arg2}
  util.return %0 : !stream.resource<staging>
}

// -----

// CHECK-LABEL: @denseTensorSplatI2
util.func public @denseTensorSplatI2(%arg0: i2, %arg1: index, %arg2: index) -> !stream.resource<*> {
  // CHECK: %[[C2:.+]] = arith.constant 2 : i8
  // CHECK: %[[PART:.+]] = arith.extui %arg0 : i2 to i8
  // CHECK: %[[SHL0:.+]] = arith.shli %[[PART]], %[[C2]] : i8
  // CHECK: %[[OR0:.+]] = arith.ori %[[SHL0]], %[[PART]] : i8
  // CHECK: %[[SHL1:.+]] = arith.shli %[[OR0]], %[[C2]] : i8
  // CHECK: %[[OR1:.+]] = arith.ori %[[SHL1]], %[[PART]] : i8
  // CHECK: %[[SH2:.+]] = arith.shli %[[OR1]], %[[C2]] : i8
  // CHECK: %[[FULL:.+]] = arith.ori %[[SH2]], %[[PART]] : i8
  // CHECK: %[[SPLAT:.+]] = stream.async.splat %[[FULL]] : i8 -> !stream.resource<*>{%arg2}
  %0 = stream.tensor.splat %arg0 : i2 -> tensor<?x1x16xi2>{%arg1} in !stream.resource<*>{%arg2}
  // CHECK: util.return %[[SPLAT]] : !stream.resource<*>
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorFillI4
util.func public @denseTensorFillI4(%arg0: i4, %arg1: !stream.resource<*>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) -> !stream.resource<*> {
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i8
  // CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
  // CHECK: %[[PART:.+]] = arith.extui %arg0 : i4 to i8
  // CHECK: %[[SHL:.+]] = arith.shli %[[PART]], %[[C4]] : i8
  // CHECK: %[[FULL:.+]] = arith.ori %[[SHL]], %[[PART]] : i8
  // CHECK: %[[MUL0:.+]] = arith.muli %arg4, %[[C16]] : index
  // CHECK: %[[ADD0:.+]] = arith.addi %[[MUL0]], %arg5 : index
  // CHECK: %[[OFFSET:.+]] = arith.divui %[[ADD0]], %[[C2]] : index
  // CHECK: %[[MUL1:.+]] = arith.muli %arg6, %[[C16]] : index
  // CHECK: %[[ADD1:.+]] = arith.addi %[[MUL1]], %arg7 : index
  // CHECK: %[[LEN:.+]] = arith.divui %[[ADD1]], %[[C2]] : index
  // CHECK: %[[END:.+]] = arith.addi %[[OFFSET]], %[[LEN]] : index
  // CHECK: %[[FILL:.+]] = stream.async.fill %[[FULL]], %arg1[%[[OFFSET]] to %[[END]] for %[[LEN]]] : i8 -> %arg1 as !stream.resource<*>{%arg3}
  %0 = stream.tensor.fill %arg0, %arg1[%arg4, %arg5 for %arg6, %arg7] : i4 -> tensor<?x16xi4>{%arg2} in %arg1 as !stream.resource<*>{%arg3}
  // CHECK: util.return %[[FILL]]
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorSliceI2
util.func public @denseTensorSliceI2(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> !stream.resource<*> {
  %c2 = arith.constant 2 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
  // CHECK: %[[MUL:.+]] = arith.muli %arg5, %[[C8]] : index
  // CHECK: %[[ADD:.+]] = arith.addi %[[MUL]], %arg6 : index
  // CHECK: %[[OFFSET:.+]] = arith.divui %[[ADD]], %[[C4]] : index
  // CHECK: %[[LEN:.+]] = arith.addi %[[OFFSET]], %arg4 : index
  // CHECK: %[[SLICE:.+]] = stream.async.slice %arg0[%[[OFFSET]] to %[[LEN]]] : !stream.resource<*>{%arg2} -> !stream.resource<*>{%arg4}
  %0 = stream.tensor.slice %arg0[%arg5, %arg6 for %arg3, %c2] : tensor<?x8xi2>{%arg1} in !stream.resource<*>{%arg2} -> tensor<?x2xi2>{%arg3} in !stream.resource<*>{%arg4}
  // CHECK: util.return %[[SLICE]] : !stream.resource<*>
  util.return %0 : !stream.resource<*>
}

// -----

// Ensures that a non-power-of-two type (i3) slice is expanded to a full byte
// because we don't currently do unaligned sub-byte packing.

// CHECK-LABEL: @denseTensorSliceI3
util.func public @denseTensorSliceI3(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> !stream.resource<*> {
  %c2 = arith.constant 2 : index
  // CHECK: %[[C8:.+]] = arith.constant 8 : index
  // CHECK: %[[MUL:.+]] = arith.muli %arg5, %[[C8]] : index
  // CHECK: %[[OFFSET:.+]] = arith.addi %[[MUL]], %arg6 : index
  // CHECK: %[[LEN:.+]] = arith.addi %[[OFFSET]], %arg4 : index
  // CHECK: %[[SLICE:.+]] = stream.async.slice %arg0[%[[OFFSET]] to %[[LEN]]] : !stream.resource<*>{%arg2} -> !stream.resource<*>{%arg4}
  %0 = stream.tensor.slice %arg0[%arg5, %arg6 for %arg3, %c2] : tensor<?x8xi3>{%arg1} in !stream.resource<*>{%arg2} -> tensor<?x2xi3>{%arg3} in !stream.resource<*>{%arg4}
  // CHECK: util.return %[[SLICE]] : !stream.resource<*>
  util.return %0 : !stream.resource<*>
}

// -----

// Ensures that a non-power-of-two type (i3) update is expanded to a full byte
// because we don't currently do unaligned sub-byte packing.

// CHECK-LABEL: @denseTensorUpdateI3
util.func public @denseTensorUpdateI3(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.resource<*>, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> !stream.resource<*> {
  // CHECK: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: %[[MUL:.+]] = arith.muli %arg5, %[[C4]] : index
  // CHECK: %[[OFFSET:.+]] = arith.addi %[[MUL]], %arg6 : index
  // CHECK: %[[LEN:.+]] = arith.addi %[[OFFSET]], %arg1 : index
  // CHECK: %[[UPDATE:.+]] = stream.async.update %arg0, %arg2[%[[OFFSET]] to %[[LEN]]] : !stream.resource<*>{%arg1} -> %arg2 as !stream.resource<*>{%arg4}

  %0 = stream.tensor.update %arg0, %arg2[%arg5, %arg6] : tensor<8x4xi3> in !stream.resource<*>{%arg1} -> tensor<?x4xi3>{%arg3} in %arg2 as !stream.resource<*>{%arg4}
  // CHECK: util.return %[[UPDATE]] : !stream.resource<*>
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorUpdateI4
util.func public @denseTensorUpdateI4(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.resource<*>, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> !stream.resource<*> {
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  // CHECK: %[[MUL:.+]] = arith.muli %arg5, %[[C4]] : index
  // CHECK: %[[ADD:.+]] = arith.addi %[[MUL]], %arg6 : index
  // CHECK: %[[OFFSET:.+]] = arith.divui %[[ADD]], %[[C2]] : index
  // CHECK: %[[LEN:.+]] = arith.addi %[[OFFSET]], %arg1 : index
  // CHECK: %[[UPDATE:.+]] = stream.async.update %arg0, %arg2[%[[OFFSET]] to %[[LEN]]] : !stream.resource<*>{%arg1} -> %arg2 as !stream.resource<*>{%arg4}
  %0 = stream.tensor.update %arg0, %arg2[%arg5, %arg6] : tensor<8x4xi4> in !stream.resource<*>{%arg1} -> tensor<?x4xi4>{%arg3} in %arg2 as !stream.resource<*>{%arg4}
  // CHECK: util.return %[[UPDATE]] : !stream.resource<*>
  util.return %0 : !stream.resource<*>
}

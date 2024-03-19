// RUN: iree-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @expandStaticShapeConstant
util.func public @expandStaticShapeConstant() -> (tensor<2x4xi32>, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-DAG: %[[CST:.+]] = arith.constant dense<2> : tensor<2x4xi32>
  %0 = flow.tensor.constant dense<2> : tensor<2x4xi32> -> tensor<2x4xi32>
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  %d0 = tensor.dim %0, %c0 : tensor<2x4xi32>
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  %d1 = tensor.dim %0, %c1 : tensor<2x4xi32>
  // CHECK: util.return %[[CST]], %[[C2]], %[[C4]]
  util.return %0, %d0, %d1 : tensor<2x4xi32>, index, index
}

// -----

// CHECK-LABEL: @expandDynamicShapeConstant
util.func public @expandDynamicShapeConstant() -> (tensor<?x?xi32>, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-DAG: %[[CST:.+]] = arith.constant dense<2> : tensor<2x4xi32>
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[D0:.+]] = util.optimization_barrier %[[C2]] : index
  // CHECK-DAG: %[[D1:.+]] = util.optimization_barrier %[[C4]] : index
  // CHECK: %[[T:.+]] = flow.tensor.reshape %[[CST]] : tensor<2x4xi32> -> tensor<?x?xi32>{%[[D0]], %[[D1]]}
  %0 = flow.tensor.constant dense<2> : tensor<2x4xi32> -> tensor<?x?xi32>
  %d0 = tensor.dim %0, %c0 : tensor<?x?xi32>
  %d1 = tensor.dim %0, %c1 : tensor<?x?xi32>
  // CHECK: util.return %[[T]], %[[D0]], %[[D1]]
  util.return %0, %d0, %d1 : tensor<?x?xi32>, index, index
}

// -----

// CHECK-LABEL: @tieShapeStaticZeroElements
util.func public @tieShapeStaticZeroElements(%arg0: tensor<0xi32>) -> tensor<0xi32> {
  // CHECK-NOT: flow.tensor.tie_shape
  %0 = flow.tensor.tie_shape %arg0 : tensor<0xi32>
  // CHECK: util.return %arg0
  util.return %0 : tensor<0xi32>
}

// -----

// CHECK-LABEL: @tieShapeDynamicZeroElements
// CHECK-SAME: (%[[OPERAND:.+]]: tensor<0x?xi32>, %[[DIM:.+]]: index)
util.func public @tieShapeDynamicZeroElements(%arg0: tensor<0x?xi32>, %dim: index) -> tensor<0x?xi32> {
  // CHECK-NOT: flow.tensor.tie_shape
  // CHECK: %[[RET:.+]] = flow.tensor.empty : tensor<0x?xi32>{%[[DIM]]}
  %0 = flow.tensor.tie_shape %arg0 : tensor<0x?xi32>{%dim}
  // CHECK: util.return %[[RET]]
  util.return %0 : tensor<0x?xi32>
}

// -----

// CHECK-LABEL: @reshapeNoOpScalar
util.func public @reshapeNoOpScalar(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT: util.return %arg0 : tensor<f32>
  %0 = flow.tensor.reshape %arg0 : tensor<f32> -> tensor<f32>
  util.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @reshapeNoOpStatic
util.func public @reshapeNoOpStatic(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NEXT: util.return %arg0 : tensor<4x4xf32>
  %0 = flow.tensor.reshape %arg0 : tensor<4x4xf32> -> tensor<4x4xf32>
  util.return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: @bitcastSameBitWidth
util.func public @bitcastSameBitWidth(%arg0: tensor<f32>) -> tensor<i32> {
  // CHECK-NEXT: flow.tensor.bitcast %arg0
  %0 = flow.tensor.bitcast %arg0 : tensor<f32> -> tensor<i32>
  util.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: @reshapeRankDifferent
util.func public @reshapeRankDifferent(%arg0: tensor<1xf32>) -> tensor<f32> {
  // CHECK-NEXT: flow.tensor.reshape %arg0
  %0 = flow.tensor.reshape %arg0 : tensor<1xf32> -> tensor<f32>
  util.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @reshapeStaticDifferent
util.func public @reshapeStaticDifferent(%arg0: tensor<1x4xf32>) -> tensor<4x1xf32> {
  // CHECK-NEXT: flow.tensor.reshape %arg0
  %0 = flow.tensor.reshape %arg0 : tensor<1x4xf32> -> tensor<4x1xf32>
  util.return %0 : tensor<4x1xf32>
}

// -----

// CHECK-LABEL: @reshapeNoOpDynamic
util.func public @reshapeNoOpDynamic(%arg0: tensor<4x?xf32>, %dim: index) -> tensor<4x?xf32> {
  // CHECK-NEXT: util.return %arg0 : tensor<4x?xf32>
  %0 = flow.tensor.reshape %arg0 : tensor<4x?xf32>{%dim} -> tensor<4x?xf32>{%dim}
  util.return %0 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @reshapeDynamicDifferent
util.func public @reshapeDynamicDifferent(%arg0: tensor<4x?xf32>, %dim0: index, %dim1: index) -> tensor<4x?xf32> {
  // CHECK-NEXT: flow.tensor.reshape %arg0
  %0 = flow.tensor.reshape %arg0 : tensor<4x?xf32>{%dim0} -> tensor<4x?xf32>{%dim1}
  util.return %0 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @flattenReshapeChain
// CHECK-SAME: %[[ARG:.+]]: tensor<4x?xf32>,
// CHECK-SAME: %[[DIM0:.+]]: index, %[[DIM1:.+]]: index, %[[DIM2:.+]]: index
util.func public @flattenReshapeChain(%arg0: tensor<4x?xf32>, %dim0: index, %dim1: index, %dim2: index) -> tensor<4x?xf32> {
  // CHECK-NEXT: %[[RET:.+]] = flow.tensor.reshape %[[ARG]] : tensor<4x?xf32>{%[[DIM0]]} -> tensor<4x?xf32>{%[[DIM2]]}
  %0 = flow.tensor.reshape %arg0 : tensor<4x?xf32>{%dim0} -> tensor<4x?xf32>{%dim1}
  %1 = flow.tensor.reshape %0 : tensor<4x?xf32>{%dim1} -> tensor<4x?xf32>{%dim2}
  // CHECK-NEXT: util.return %[[RET]]
  util.return %1 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @flattenReshapeBitcastChain
// CHECK-SAME: %[[ARG:.+]]: tensor<4x?xi16>,
// CHECK-SAME: %[[DIM0:.+]]: index, %[[DIM1:.+]]: index, %[[DIM2:.+]]: index
util.func public @flattenReshapeBitcastChain(%arg0: tensor<4x?xi16>, %dim0: index, %dim1: index, %dim2: index) -> tensor<4x?xbf16> {
  // CHECK-NEXT: %[[RET:.+]] = flow.tensor.bitcast %[[ARG]] : tensor<4x?xi16>{%[[DIM0]]} -> tensor<4x?xbf16>{%[[DIM2]]}
  %0 = flow.tensor.bitcast %arg0 : tensor<4x?xi16>{%dim0} -> tensor<4x?xf16>{%dim1}
  %1 = flow.tensor.bitcast %0 : tensor<4x?xf16>{%dim1} -> tensor<4x?xbf16>{%dim2}
  // CHECK-NEXT: util.return %[[RET]]
  util.return %1 : tensor<4x?xbf16>
}

// -----

// CHECK-LABEL: @flattenBitCastChain
// CHECK-SAME: %[[ARG:.+]]: tensor<?x4xi16>,
// CHECK-SAME: %[[DIM0:.+]]: index, %[[DIM1:.+]]: index, %[[DIM2:.+]]: index
util.func public @flattenBitCastChain(%arg0: tensor<?x4xi16>, %dim0: index, %dim1: index, %dim2: index) -> tensor<?x8xi8> {
  // CHECK-NEXT: %[[RET:.+]] = flow.tensor.bitcast %[[ARG]] : tensor<?x4xi16>{%[[DIM0]]} -> tensor<?x8xi8>{%[[DIM2]]}
  %0 = flow.tensor.bitcast %arg0 : tensor<?x4xi16>{%dim0} -> tensor<?x2xi32>{%dim1}
  %1 = flow.tensor.bitcast %0 : tensor<?x2xi32>{%dim1} -> tensor<?x8xi8>{%dim2}
  // CHECK-NEXT: util.return %[[RET]]
  util.return %1 : tensor<?x8xi8>
}

// -----

// CHECK-LABEL: @flattenBitCastReshapeBitCast
// CHECK-SAME: %[[ARG:.+]]: tensor<?x16xi16>,
// CHECK-SAME: %[[DIM0:.+]]: index, %[[DIM1:.+]]: index, %[[DIM2:.+]]: index, %[[DIM3:.+]]: index
util.func public @flattenBitCastReshapeBitCast(%arg0: tensor<?x16xi16>, %dim0: index, %dim1: index, %dim2: index, %dim3: index) -> tensor<?x4x4xi16> {
  // CHECK-NEXT: %[[RET:.+]] = flow.tensor.reshape %[[ARG]] : tensor<?x16xi16>{%[[DIM0]]} -> tensor<?x4x4xi16>{%[[DIM3]]}
  %0 = flow.tensor.bitcast %arg0 : tensor<?x16xi16>{%dim0} -> tensor<?x8xi32>{%dim1}
  %1 = flow.tensor.reshape %0 : tensor<?x8xi32>{%dim1} -> tensor<?x4x2xi32>{%dim2}
  %2 = flow.tensor.bitcast %1 : tensor<?x4x2xi32>{%dim2} -> tensor<?x4x4xi16>{%dim3}
  // CHECK-NEXT: util.return %[[RET]]
  util.return %2 : tensor<?x4x4xi16>
}


// -----

// CHECK-LABEL: @reshapeFromStaticZeroElements
// CHECK-SAME: (%[[OPERAND:.+]]: tensor<4x0xf32>, %[[DIM:.+]]: index)
util.func public @reshapeFromStaticZeroElements(%arg0: tensor<4x0xf32>, %dim: index) -> tensor<4x?xf32> {
  // CHECK: %[[RET:.+]] = flow.tensor.empty : tensor<4x?xf32>{%[[DIM]]}
  %0 = flow.tensor.reshape %arg0 : tensor<4x0xf32> -> tensor<4x?xf32>{%dim}
  // CHECK-NEXT: util.return %[[RET]]
  util.return %0 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @reshapeFromDynamicZeroElements
// CHECK-SAME: (%[[OPERAND:.+]]: tensor<0x?xf32>, %[[DIM0:.+]]: index, %[[DIM1:.+]]: index)
util.func public @reshapeFromDynamicZeroElements(%arg0: tensor<0x?xf32>, %dim0: index, %dim1: index) -> tensor<4x?xf32> {
  // CHECK: %[[RET:.+]] = flow.tensor.empty : tensor<4x?xf32>{%[[DIM1]]}
  %0 = flow.tensor.reshape %arg0 : tensor<0x?xf32>{%dim0} -> tensor<4x?xf32>{%dim1}
  // CHECK-NEXT: util.return %[[RET]]
  util.return %0 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @reshapeToStaticZeroElements
util.func public @reshapeToStaticZeroElements(%arg0: tensor<4x?xf32>, %dim0: index) {
  // CHECK-NEXT: %[[RET:.+]] = flow.tensor.empty : tensor<4x0xf32>
  %0 = flow.tensor.reshape %arg0 : tensor<4x?xf32>{%dim0} -> tensor<4x0xf32>
  // CHECK-NEXT: util.optimization_barrier %[[RET]]
  util.optimization_barrier %0 : tensor<4x0xf32>
  util.return
}

// -----

// CHECK-LABEL: @reshapeToDynamicZeroElements
// CHECK-SAME: (%[[OPERAND:.+]]: tensor<4x?xf32>, %[[DIM0:.+]]: index, %[[DIM1:.+]]: index)
util.func public @reshapeToDynamicZeroElements(%arg0: tensor<4x?xf32>, %dim0: index, %dim1: index) {
  // CHECK: %[[RET:.+]] = flow.tensor.empty : tensor<0x?xf32>{%[[DIM1]]}
  %0 = flow.tensor.reshape %arg0 : tensor<4x?xf32>{%dim0} -> tensor<0x?xf32>{%dim1}
  // CHECK-NEXT: util.optimization_barrier %[[RET]]
  util.optimization_barrier %0 : tensor<0x?xf32>
  util.return
}

// -----

// CHECK-LABEL: @reshapeEmpty
// CHECK-SAME: (%[[DIM:.+]]: index)
util.func public @reshapeEmpty(%dim: index) -> tensor<?xi32> {
  // CHECK: %[[RET:.+]] = flow.tensor.empty : tensor<?xi32>{%[[DIM]]}
  %0 = flow.tensor.empty : tensor<1x?xi32>{%dim}
  // CHECK-NOT: flow.tensor.reshape
  %1 = flow.tensor.reshape %0 : tensor<1x?xi32>{%dim} -> tensor<?xi32>{%dim}
  // CHECK: util.return %[[RET]]
  util.return %1 : tensor<?xi32>
}

// -----

// CHECK-LABEL: @loadConst
util.func public @loadConst() -> i32 {
  %0 = arith.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NEXT: %[[C2:.+]] = arith.constant 2 : i32
  %2 = flow.tensor.load %0[%c1, %c0] : tensor<2x2xi32>
  // CHECK-NEXT: util.return %[[C2]]
  util.return %2 : i32
}

// -----

// CHECK-LABEL: @loadConstScalar
util.func public @loadConstScalar() -> i32 {
  %0 = arith.constant dense<4> : tensor<i32>
  // CHECK-NEXT: %[[C4:.+]] = arith.constant 4 : i32
  %1 = flow.tensor.load %0 : tensor<i32>
  // CHECK-NEXT: util.return %[[C4]]
  util.return %1 : i32
}

// -----

// CHECK-LABEL: @storeConst
util.func public @storeConst() -> tensor<2x2xi32> {
  %0 = arith.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<[
  // CHECK-SAME:     [0, 1], [4, 3]
  // CHECK-SAME: ]> : tensor<2x2xi32>
  %1 = flow.tensor.store %c4, %0[%c1, %c0] : tensor<2x2xi32>
  // CHECK-NEXT: util.return %[[C]]
  util.return %1 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: @storeConstScalar
util.func public @storeConstScalar() -> tensor<i32> {
  %0 = arith.constant dense<0> : tensor<i32>
  %1 = arith.constant 4 : i32
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<4> : tensor<i32>
  %2 = flow.tensor.store %1, %0 : tensor<i32>
  // CHECK-NEXT: util.return %[[C]]
  util.return %2 : tensor<i32>
}

// -----

// CHECK-LABEL: @allocaDims
//  CHECK-SAME: (%[[DIM:.+]]: index)
util.func public @allocaDims(%dim: index) -> (index, index, index) {
  // CHECK-NOT: flow.tensor.alloca
  %0 = flow.tensor.alloca : tensor<4x?x0xf32>{%dim}
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %0, %c0 : tensor<4x?x0xf32>
  %d1 = tensor.dim %0, %c1 : tensor<4x?x0xf32>
  %d2 = tensor.dim %0, %c2 : tensor<4x?x0xf32>
  // CHECK: util.return %c4, %[[DIM]], %c0
  util.return %d0, %d1, %d2 : index, index, index
}

// -----

// CHECK-LABEL: @emptyDims
//  CHECK-SAME: (%[[DIM:.+]]: index)
util.func public @emptyDims(%dim: index) -> (index, index, index) {
  // CHECK-NOT: flow.tensor.empty
  %0 = flow.tensor.empty : tensor<4x?x0xf32>{%dim}
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %0, %c0 : tensor<4x?x0xf32>
  %d1 = tensor.dim %0, %c1 : tensor<4x?x0xf32>
  %d2 = tensor.dim %0, %c2 : tensor<4x?x0xf32>
  // CHECK: util.return %c4, %[[DIM]], %c0
  util.return %d0, %d1, %d2 : index, index, index
}

// -----

// CHECK-LABEL: @splatDynamicShape
//  CHECK-SAME: (%[[DIM0:.+]]: index, %[[DIM1:.+]]: index)
util.func public @splatDynamicShape(%dim0: index, %dim1: index) -> tensor<?x?xi32> {
  // CHECK: %[[FOUR:.+]] = arith.constant 4 : i32
  %four = arith.constant 4 : i32
  // CHECK: %[[SPLAT:.+]] = flow.tensor.splat %[[FOUR]] : tensor<?x?xi32>{%[[DIM0]], %[[DIM1]]}
  %1 = flow.tensor.splat %four : tensor<?x?xi32>{%dim0, %dim1}
  // CHECK: util.return %[[SPLAT]]
  util.return %1 : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: @splatStaticZeroElements
util.func public @splatStaticZeroElements(%value: f32) -> tensor<0x2xf32> {
  // CHECK-NEXT: %[[RET:.+]] = flow.tensor.empty : tensor<0x2xf32>
  %0 = flow.tensor.splat %value : tensor<0x2xf32>
  // CHECK-NEXT: util.return %[[RET]]
  util.return %0 : tensor<0x2xf32>
}

// -----

// CHECK-LABEL: @splatDynamicZeroElements
//  CHECK-SAME: (%[[VALUE:.+]]: f32, %[[DIM:.+]]: index)
util.func public @splatDynamicZeroElements(%value: f32, %dim: index) -> tensor<0x?xf32> {
  // CHECK-NEXT: %[[RET:.+]] = flow.tensor.empty : tensor<0x?xf32>{%[[DIM]]}
  %0 = flow.tensor.splat %value : tensor<0x?xf32>{%dim}
  // CHECK-NEXT: util.return %[[RET]]
  util.return %0 : tensor<0x?xf32>
}

// -----

// CHECK-LABEL: @cloneConst
util.func public @cloneConst() -> tensor<4xi32> {
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %0 = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %1 = flow.tensor.clone %0 : tensor<4xi32>
  // CHECK-NEXT: util.return %[[C]]
  util.return %1 : tensor<4xi32>
}

// -----

// CHECK-LABEL: @cloneConstZeroElements
util.func public @cloneConstZeroElements() -> tensor<0x2xi32> {
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<> : tensor<0x2xi32>
  %0 = arith.constant dense<> : tensor<0x2xi32>
  // CHECK-NOT: flow.tensor.clone
  %1 = flow.tensor.clone %0 : tensor<0x2xi32>
  // CHECK-NEXT: util.return %[[C]]
  util.return %1 : tensor<0x2xi32>
}

// -----

// CHECK-LABEL: @cloneStaticZeroElements
util.func public @cloneStaticZeroElements(%arg0: tensor<0x2xf32>) -> tensor<0x2xf32> {
  // CHECK-NOT: flow.tensor.clone
  // CHECK-NEXT: %[[RET:.+]] = flow.tensor.empty : tensor<0x2xf32>
  %0 = flow.tensor.clone %arg0 : tensor<0x2xf32>
  // CHECK-NEXT: %[[RET]]
  util.return %0 : tensor<0x2xf32>
}

// -----

// CHECK-LABEL: @cloneDynamicZeroElements
//  CHECK-SAME: (%[[OPERAND:.+]]: tensor<0x?xf32>, %[[DIM:.+]]: index)
util.func public @cloneDynamicZeroElements(%arg0: tensor<0x?xf32>, %dim: index) -> tensor<0x?xf32> {
  // CHECK-NOT: flow.tensor.clone
  // CHECK-NEXT: %[[RET:.+]] = flow.tensor.empty : tensor<0x?xf32>{%[[DIM]]}
  %0 = flow.tensor.clone %arg0 : tensor<0x?xf32>{%dim}
  // CHECK-NEXT: %[[RET]]
  util.return %0 : tensor<0x?xf32>
}

// -----

// CHECK-LABEL: @sliceConst0D
util.func public @sliceConst0D() -> tensor<i32> {
  %0 = arith.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<0> : tensor<i32>
  %1 = flow.tensor.slice %0[for] : tensor<i32> -> tensor<i32>
  // CHECK-NEXT: util.return %[[C]]
  util.return %1 : tensor<i32>
}

// -----

// CHECK-LABEL: @sliceConst1D
util.func public @sliceConst1D() -> tensor<1xi32> {
  %0 = arith.constant dense<0> : tensor<1xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<0> : tensor<1xi32>
  %1 = flow.tensor.slice %0[%c0 for %c1] : tensor<1xi32> -> tensor<1xi32>
  // CHECK-NEXT: util.return %[[C]]
  util.return %1 : tensor<1xi32>
}

// -----

// CHECK-LABEL: @sliceConst1DZeroLength
util.func public @sliceConst1DZeroLength() -> tensor<0xi32> {
  %0 = arith.constant dense<0> : tensor<1xi32>
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<> : tensor<0xi32>
  %1 = flow.tensor.slice %0[%c0 for %c0] : tensor<1xi32> -> tensor<0xi32>
  // CHECK-NEXT: util.return %[[C]]
  util.return %1 : tensor<0xi32>
}

// -----

// CHECK-LABEL: @sliceConst2D
util.func public @sliceConst2D() -> tensor<1x2xi32> {
  %0 = arith.constant dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<[
  // CHECK-SAME:     [1, 2]
  // CHECK-SAME: ]> : tensor<1x2xi32>
  %1 = flow.tensor.slice %0[%c0, %c1 for %c1, %c2] : tensor<2x3xi32> -> tensor<1x2xi32>
  // CHECK-NEXT: util.return %[[C]]
  util.return %1 : tensor<1x2xi32>
}

// -----

// CHECK-LABEL: @sliceConst2DZeroLength1
util.func public @sliceConst2DZeroLength1() -> tensor<1x0xi32> {
  %0 = arith.constant dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<> : tensor<1x0xi32>
  %1 = flow.tensor.slice %0[%c0, %c0 for %c1, %c0] : tensor<2x3xi32> -> tensor<1x0xi32>
  // CHECK-NEXT: util.return %[[C]]
  util.return %1 : tensor<1x0xi32>
}

// -----

// CHECK-LABEL: @sliceConst2DZeroLength01
util.func public @sliceConst2DZeroLength01() -> tensor<0x0xi32> {
  %0 = arith.constant dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi32>
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<> : tensor<0x0xi32>
  %1 = flow.tensor.slice %0[%c0, %c0 for %c0, %c0] : tensor<2x3xi32> -> tensor<0x0xi32>
  // CHECK-NEXT: util.return %[[C]]
  util.return %1 : tensor<0x0xi32>
}

// -----

// CHECK-LABEL: @sliceFromZeroElements
util.func public @sliceFromZeroElements(%arg0: tensor<0xi32>) -> tensor<?xi32> {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: flow.tensor.slice
  // CHECK: %[[RET:.+]] = flow.tensor.empty : tensor<?xi32>{%c0}
  %0 = flow.tensor.slice %arg0[%c0 for %c0] : tensor<0xi32> -> tensor<?xi32>{%c0}
  // CHECK: util.return %[[RET]]
  util.return %0 : tensor<?xi32>
}

// -----

// CHECK-LABEL: @sliceZeroElements
util.func public @sliceZeroElements(%arg0: tensor<?xi32>, %dim: index) -> tensor<0xi32> {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: flow.tensor.slice
  // CHECK: %[[RET:.+]] = flow.tensor.empty : tensor<0xi32>
  %0 = flow.tensor.slice %arg0[%c0 for %c0] : tensor<?xi32>{%dim} -> tensor<0xi32>
  // CHECK: util.return %[[RET]]
  util.return %0 : tensor<0xi32>
}

// -----

// CHECK-LABEL: @sliceConst3D
util.func public @sliceConst3D() -> tensor<1x2x3xi32> {
  %0 = arith.constant dense<[[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[9, 10, 11], [12, 13, 14], [15, 16, 17]]]> : tensor<2x3x3xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<[
  // CHECK-SAME:                             [
  // CHECK-SAME:                              [3, 4, 5], [6, 7, 8]]]> : tensor<1x2x3xi32>
  %1 = flow.tensor.slice %0[%c0, %c1, %c0 for %c1, %c2, %c3] : tensor<2x3x3xi32> -> tensor<1x2x3xi32>
  // CHECK-NEXT: util.return %[[C]]
  util.return %1 : tensor<1x2x3xi32>
}

// -----

// CHECK-LABEL: @updateConst0D
util.func public @updateConst0D() -> tensor<i32> {
  %0 = arith.constant dense<0> : tensor<i32>
  %1 = arith.constant dense<1> : tensor<i32>
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<0> : tensor<i32>
  %2 = flow.tensor.update %0, %1[] : tensor<i32> -> tensor<i32>
  // CHECK-NEXT: util.return %[[C]]
  util.return %2 : tensor<i32>
}

// -----

// CHECK-LABEL: @updateConst1D
util.func public @updateConst1D() -> tensor<1xi32> {
  %0 = arith.constant dense<0> : tensor<1xi32>
  %1 = arith.constant dense<1> : tensor<1xi32>
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<0> : tensor<1xi32>
  %2 = flow.tensor.update %0, %1[%c0] : tensor<1xi32> -> tensor<1xi32>
  // CHECK-NEXT: util.return %[[C]]
  util.return %2 : tensor<1xi32>
}

// -----

// CHECK-LABEL: @updateConst1DUpdateZeroSize
util.func public @updateConst1DUpdateZeroSize() -> tensor<1xi32> {
  %0 = arith.constant dense<> : tensor<0xi32>
  %1 = arith.constant dense<1> : tensor<1xi32>
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<1> : tensor<1xi32>
  %2 = flow.tensor.update %0, %1[%c0] : tensor<0xi32> -> tensor<1xi32>
  // CHECK-NEXT: util.return %[[C]]
  util.return %2 : tensor<1xi32>
}

// -----

// CHECK-LABEL: @updateConst2DUpdate1x1
util.func public @updateConst2DUpdate1x1() -> tensor<3x4xi32> {
  %0 = arith.constant dense<[[12]]> : tensor<1x1xi32>
  %1 = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]> : tensor<3x4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<[
  // CHECK-SAME: [0, 12, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]> : tensor<3x4xi32>
  %2 = flow.tensor.update %0, %1[%c0, %c1] : tensor<1x1xi32> -> tensor<3x4xi32>
  // CHECK-NEXT: util.return %[[C]]
  util.return %2 : tensor<3x4xi32>
}

// -----

// CHECK-LABEL: @updateConst2DUpdate2x2
util.func public @updateConst2DUpdate2x2() -> tensor<3x4xi32> {
  %0 = arith.constant dense<[[12, 13], [14, 15]]> : tensor<2x2xi32>
  %1 = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]> : tensor<3x4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<[
  // CHECK-SAME: [0, 12, 13, 3], [4, 14, 15, 7], [8, 9, 10, 11]]> : tensor<3x4xi32>
  %2 = flow.tensor.update %0, %1[%c0, %c1] : tensor<2x2xi32> -> tensor<3x4xi32>
  // CHECK-NEXT: util.return %[[C]]
  util.return %2 : tensor<3x4xi32>
}

// -----

// CHECK-LABEL: @updateConst3DUpdate1x2x3
util.func public @updateConst3DUpdate1x2x3() -> tensor<2x3x3xi32> {
  %0 = arith.constant dense<[[[18, 19, 20], [21, 22, 23]]]> : tensor<1x2x3xi32>
  %1 = arith.constant dense<[[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[9, 10, 11], [12, 13, 14], [15, 16, 17]]]> : tensor<2x3x3xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<[
  // CHECK-SAME:                             [
  // CHECK-SAME:                              [0, 1, 2], [18, 19, 20], [21, 22, 23]], [
  // CHECK-SAME: [9, 10, 11], [12, 13, 14], [15, 16, 17]]]> : tensor<2x3x3xi32>
  %2 = flow.tensor.update %0, %1[%c0, %c1, %c0] : tensor<1x2x3xi32> -> tensor<2x3x3xi32>
  // CHECK-NEXT: util.return %[[C]]
  util.return %2 : tensor<2x3x3xi32>
}

// -----

// CHECK-LABEL: @updateConst3DUpdate2x3x2
util.func public @updateConst3DUpdate2x3x2() -> tensor<2x3x3xi32> {
  %0 = arith.constant dense<[[[18, 19], [20, 21], [22, 23]], [[24, 25], [26, 27], [28, 29]]]> : tensor<2x3x2xi32>
  %1 = arith.constant dense<[[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[9, 10, 11], [12, 13, 14], [15, 16, 17]]]> : tensor<2x3x3xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 0 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<[
  // CHECK-SAME:                             [
  // CHECK-SAME:                              [18, 19, 2], [20, 21, 5], [22, 23, 8]], [
  // CHECK-SAME: [24, 25, 11], [26, 27, 14], [28, 29, 17]]]> : tensor<2x3x3xi32>
  %2 = flow.tensor.update %0, %1[%c0, %c1, %c0] : tensor<2x3x2xi32> -> tensor<2x3x3xi32>
  // CHECK-NEXT: util.return %[[C]]
  util.return %2 : tensor<2x3x3xi32>
}

// -----

// CHECK-LABEL: @updateReplace
util.func public @updateReplace(%arg0 : tensor<4xi32>, %arg1 : tensor<4xi32>) -> tensor<4xi32> {
  %c0 = arith.constant 0 : index
  %0 = flow.tensor.update %arg0, %arg1[%c0] : tensor<4xi32> -> tensor<4xi32>
  // CHECK-NEXT: util.return %arg0
  util.return %0 : tensor<4xi32>
}

// -----

// CHECK-LABEL: @updateIntoZeroElements
util.func public @updateIntoZeroElements(%update: tensor<?x?xi32>, %dim: index, %target: tensor<0x0xi32>) -> tensor<0x0xi32> {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: flow.tensor.update
  // CHECK-NEXT: %[[RET:.+]] = flow.tensor.empty : tensor<0x0xi32>
  %0 = flow.tensor.update %update, %target[%c0, %c0] : tensor<?x?xi32>{%dim, %dim} -> tensor<0x0xi32>
  // CHECK-NEXT: util.return %[[RET]]
  util.return %0 : tensor<0x0xi32>
}

// -----

// CHECK-LABEL: @updateZeroElements
//  CHECK-SAME: (%[[UPDATE:.+]]: tensor<0x1xi32>, %[[TARGET:.+]]: tensor<1x1xi32>)
util.func public @updateZeroElements(%update: tensor<0x1xi32>, %target: tensor<1x1xi32>) -> tensor<1x1xi32> {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: flow.tensor.update
  %0 = flow.tensor.update %update, %target[%c0, %c0] : tensor<0x1xi32> -> tensor<1x1xi32>
  // CHECK: util.return %[[TARGET]]
  util.return %0 : tensor<1x1xi32>
}

// -----

// CHECK-LABEL: @propogateStaticShapeOfTarget
util.func public @propogateStaticShapeOfTarget(%arg0 : tensor<?x?xf32>, %arg1 : f32) -> tensor<?x?xf32> {
  %c21 = arith.constant 21 : index
  %c42 = arith.constant 42 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  // CHECK: %[[TARGET:.+]] = tensor.generate {
  // CHECK: } : tensor<21x42xf32>
  %0 = tensor.generate %c21, %c42 {
  ^bb0(%arg2: index, %arg3: index):
    tensor.yield %arg1 : f32
  } :  tensor<?x?xf32>
  // CHECK: %[[UPDATED:.+]] = flow.tensor.update %{{.+}}, %[[TARGET]]
  // CHECK: %[[RESULT:.+]] = tensor.cast %[[UPDATED]] : tensor<21x42xf32> to tensor<?x?xf32>
  %1 = flow.tensor.update %arg0, %0[%c2, %c4] : tensor<?x?xf32>{%c21, %c42} -> tensor<?x?xf32>{%c21, %c42}
  // CHECK: util.return %[[RESULT]]
  util.return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @propogateStaticShapeOfUpdate
util.func public @propogateStaticShapeOfUpdate(%arg0 : tensor<?x?xf32>, %arg1 : f32) -> tensor<?x?xf32> {
  %c21 = arith.constant 21 : index
  %c42 = arith.constant 42 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  // CHECK: %[[UPDATE:.+]] = tensor.generate {
  // CHECK: } : tensor<21x42xf32>
  %0 = tensor.generate %c21, %c42 {
  ^bb0(%arg2: index, %arg3: index):
    tensor.yield %arg1 : f32
  } :  tensor<?x?xf32>
  // CHECK: %[[RESULT:.+]] = flow.tensor.update  %[[UPDATE]]
  %1 = flow.tensor.update %0, %arg0[%c2, %c4] : tensor<?x?xf32>{%c21, %c42} -> tensor<?x?xf32>{%c21, %c42}
  // CHECK: util.return %[[RESULT]]
  util.return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @foldSplatLoadIntoPrimitive
//  CHECK-SAME: (%[[arg0:.+]]: f32, %[[arg1:.+]]: index, %[[arg2:.+]]: index)
util.func public @foldSplatLoadIntoPrimitive(%arg0 : f32, %arg1 : index, %arg2 : index) -> f32 {
  // CHECK-NEXT: util.return %[[arg0]] : f32
  %0 = flow.tensor.splat %arg0 : tensor<4x4xf32>
  %1 = flow.tensor.load %0[%arg1, %arg2] : tensor<4x4xf32>
  util.return %1 : f32
}

// -----

// CHECK-LABEL: @foldSplatReshapeIntoSplat
util.func public @foldSplatReshapeIntoSplat(%arg0 : f32) -> tensor<16xf32> {
  // CHECK-NEXT: %0 = flow.tensor.splat %arg0 : tensor<16xf32>
  // CHECK-NEXT: util.return %0 : tensor<16xf32>
  %0 = flow.tensor.splat %arg0 : tensor<4x4xf32>
  %1 = flow.tensor.reshape %0 : tensor<4x4xf32> -> tensor<16xf32>
  util.return %1 : tensor<16xf32>
}

// -----

// CHECK-LABEL: @foldSplatReshapeIntoSplatDynamic
util.func public @foldSplatReshapeIntoSplatDynamic(%arg0 : f32, %arg1 : index, %arg2 : index, %arg3 : index) -> tensor<?x?xf32> {
  // CHECK-NEXT: %0 = flow.tensor.splat %arg0 : tensor<?x?xf32>{%arg2, %arg3}
  // CHECK-NEXT: util.return %0 : tensor<?x?xf32>
  %0 = flow.tensor.splat %arg0 : tensor<?x4xf32>{%arg1}
  %1 = flow.tensor.reshape %0 : tensor<?x4xf32>{%arg1} -> tensor<?x?xf32>{%arg2, %arg3}
  util.return %1 : tensor<?x?xf32>
}

// -----

util.func public @innermost_unit_dim(%4: !flow.dispatch.tensor<readonly:tensor<3x1x16x257x88xf16>>,
    %arg0: index, %arg2 : index, %10 : index, %9 : index) -> tensor<?x?x?xf16> {
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %11 = flow.dispatch.tensor.load %4, offsets = [1, 0, %arg0, %10, %arg2], sizes = [1, 1, %c16, %9, %c1], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x1x16x257x88xf16>> -> tensor<?x?x?xf16>
  util.return %11 : tensor<?x?x?xf16>
}
// CHECK-LABEL: util.func public @innermost_unit_dim
//  CHECK-SAME:     %[[DYNAMIC_DIM:[a-zA-Z0-9]+]]: index)
//       CHECK:   flow.dispatch.tensor.load
//  CHECK-SAME:       sizes = [1, 1, 16, %[[DYNAMIC_DIM]], 1]

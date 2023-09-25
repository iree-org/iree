// RUN: iree-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @expandStaticShapeConstant
func.func @expandStaticShapeConstant() -> (tensor<2x4xi32>, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-DAG: %[[CST:.+]] = arith.constant dense<2> : tensor<2x4xi32>
  %0 = flow.tensor.constant dense<2> : tensor<2x4xi32> -> tensor<2x4xi32>
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  %d0 = tensor.dim %0, %c0 : tensor<2x4xi32>
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  %d1 = tensor.dim %0, %c1 : tensor<2x4xi32>
  // CHECK: return %[[CST]], %[[C2]], %[[C4]]
  return %0, %d0, %d1 : tensor<2x4xi32>, index, index
}

// -----

// CHECK-LABEL: @expandDynamicShapeConstant
func.func @expandDynamicShapeConstant() -> (tensor<?x?xi32>, index, index) {
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
  // CHECK: return %[[T]], %[[D0]], %[[D1]]
  return %0, %d0, %d1 : tensor<?x?xi32>, index, index
}

// -----

// CHECK-LABEL: @tieShapeStaticZeroElements
func.func @tieShapeStaticZeroElements(%arg0: tensor<0xi32>) -> tensor<0xi32> {
  // CHECK-NOT: flow.tensor.tie_shape
  %0 = flow.tensor.tie_shape %arg0 : tensor<0xi32>
  // CHECK: return %arg0
  return %0 : tensor<0xi32>
}

// -----

// CHECK-LABEL: @tieShapeDynamicZeroElements
// CHECK-SAME: (%[[OPERAND:.+]]: tensor<0x?xi32>, %[[DIM:.+]]: index)
func.func @tieShapeDynamicZeroElements(%arg0: tensor<0x?xi32>, %dim: index) -> tensor<0x?xi32> {
  // CHECK-NOT: flow.tensor.tie_shape
  // CHECK: %[[RET:.+]] = flow.tensor.empty : tensor<0x?xi32>{%[[DIM]]}
  %0 = flow.tensor.tie_shape %arg0 : tensor<0x?xi32>{%dim}
  // CHECK: return %[[RET]]
  return %0 : tensor<0x?xi32>
}

// -----

// CHECK-LABEL: @reshapeNoOpScalar
func.func @reshapeNoOpScalar(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT: return %arg0 : tensor<f32>
  %0 = flow.tensor.reshape %arg0 : tensor<f32> -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @reshapeNoOpStatic
func.func @reshapeNoOpStatic(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NEXT: return %arg0 : tensor<4x4xf32>
  %0 = flow.tensor.reshape %arg0 : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: @reshapeElementTypeDifferent
func.func @reshapeElementTypeDifferent(%arg0: tensor<f32>) -> tensor<i32> {
  // CHECK-NEXT: flow.tensor.reshape %arg0
  %0 = flow.tensor.reshape %arg0 : tensor<f32> -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: @reshapeRankDifferent
func.func @reshapeRankDifferent(%arg0: tensor<1xf32>) -> tensor<f32> {
  // CHECK-NEXT: flow.tensor.reshape %arg0
  %0 = flow.tensor.reshape %arg0 : tensor<1xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @reshapeStaticDifferent
func.func @reshapeStaticDifferent(%arg0: tensor<1x4xf32>) -> tensor<4x1xf32> {
  // CHECK-NEXT: flow.tensor.reshape %arg0
  %0 = flow.tensor.reshape %arg0 : tensor<1x4xf32> -> tensor<4x1xf32>
  return %0 : tensor<4x1xf32>
}

// -----

// CHECK-LABEL: @reshapeNoOpDynamic
func.func @reshapeNoOpDynamic(%arg0: tensor<4x?xf32>, %dim: index) -> tensor<4x?xf32> {
  // CHECK-NEXT: return %arg0 : tensor<4x?xf32>
  %0 = flow.tensor.reshape %arg0 : tensor<4x?xf32>{%dim} -> tensor<4x?xf32>{%dim}
  return %0 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @reshapeDynamicDifferent
func.func @reshapeDynamicDifferent(%arg0: tensor<4x?xf32>, %dim0: index, %dim1: index) -> tensor<4x?xf32> {
  // CHECK-NEXT: flow.tensor.reshape %arg0
  %0 = flow.tensor.reshape %arg0 : tensor<4x?xf32>{%dim0} -> tensor<4x?xf32>{%dim1}
  return %0 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @flattenReshapeChain
// CHECK-SAME: %[[ARG:.+]]: tensor<4x?xf32>,
// CHECK-SAME: %[[DIM0:.+]]: index, %[[DIM1:.+]]: index, %[[DIM2:.+]]: index
func.func @flattenReshapeChain(%arg0: tensor<4x?xf32>, %dim0: index, %dim1: index, %dim2: index) -> tensor<4x?xf32> {
  // CHECK-NEXT: %[[RET:.+]] = flow.tensor.reshape %[[ARG]] : tensor<4x?xf32>{%[[DIM0]]} -> tensor<4x?xf32>{%[[DIM2]]}
  %0 = flow.tensor.reshape %arg0 : tensor<4x?xf32>{%dim0} -> tensor<4x?xf32>{%dim1}
  %1 = flow.tensor.reshape %0 : tensor<4x?xf32>{%dim1} -> tensor<4x?xf32>{%dim2}
  // CHECK-NEXT: return %[[RET]]
  return %1 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @flattenReshapeBitcastChain
// CHECK-SAME: %[[ARG:.+]]: tensor<4x?xi16>,
// CHECK-SAME: %[[DIM0:.+]]: index, %[[DIM1:.+]]: index, %[[DIM2:.+]]: index
func.func @flattenReshapeBitcastChain(%arg0: tensor<4x?xi16>, %dim0: index, %dim1: index, %dim2: index) -> tensor<4x?xbf16> {
  // CHECK-NEXT: %[[RET:.+]] = flow.tensor.reshape %[[ARG]] : tensor<4x?xi16>{%[[DIM0]]} -> tensor<4x?xbf16>{%[[DIM2]]}
  %0 = flow.tensor.reshape %arg0 : tensor<4x?xi16>{%dim0} -> tensor<4x?xf16>{%dim1}
  %1 = flow.tensor.reshape %0 : tensor<4x?xf16>{%dim1} -> tensor<4x?xbf16>{%dim2}
  // CHECK-NEXT: return %[[RET]]
  return %1 : tensor<4x?xbf16>
}

// -----

// CHECK-LABEL: @reshapeFromStaticZeroElements
// CHECK-SAME: (%[[OPERAND:.+]]: tensor<4x0xf32>, %[[DIM:.+]]: index)
func.func @reshapeFromStaticZeroElements(%arg0: tensor<4x0xf32>, %dim: index) -> tensor<4x?xf32> {
  // CHECK: %[[RET:.+]] = flow.tensor.empty : tensor<4x?xf32>{%[[DIM]]}
  %0 = flow.tensor.reshape %arg0 : tensor<4x0xf32> -> tensor<4x?xf32>{%dim}
  // CHECK-NEXT: return %[[RET]]
  return %0 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @reshapeFromDynamicZeroElements
// CHECK-SAME: (%[[OPERAND:.+]]: tensor<0x?xf32>, %[[DIM0:.+]]: index, %[[DIM1:.+]]: index)
func.func @reshapeFromDynamicZeroElements(%arg0: tensor<0x?xf32>, %dim0: index, %dim1: index) -> tensor<4x?xf32> {
  // CHECK: %[[RET:.+]] = flow.tensor.empty : tensor<4x?xf32>{%[[DIM1]]}
  %0 = flow.tensor.reshape %arg0 : tensor<0x?xf32>{%dim0} -> tensor<4x?xf32>{%dim1}
  // CHECK-NEXT: return %[[RET]]
  return %0 : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @reshapeToStaticZeroElements
func.func @reshapeToStaticZeroElements(%arg0: tensor<4x?xf32>, %dim0: index) {
  // CHECK-NEXT: %[[RET:.+]] = flow.tensor.empty : tensor<4x0xf32>
  %0 = flow.tensor.reshape %arg0 : tensor<4x?xf32>{%dim0} -> tensor<4x0xf32>
  // CHECK-NEXT: util.optimization_barrier %[[RET]]
  util.optimization_barrier %0 : tensor<4x0xf32>
  return
}

// -----

// CHECK-LABEL: @reshapeToDynamicZeroElements
// CHECK-SAME: (%[[OPERAND:.+]]: tensor<4x?xf32>, %[[DIM0:.+]]: index, %[[DIM1:.+]]: index)
func.func @reshapeToDynamicZeroElements(%arg0: tensor<4x?xf32>, %dim0: index, %dim1: index) {
  // CHECK: %[[RET:.+]] = flow.tensor.empty : tensor<0x?xf32>{%[[DIM1]]}
  %0 = flow.tensor.reshape %arg0 : tensor<4x?xf32>{%dim0} -> tensor<0x?xf32>{%dim1}
  // CHECK-NEXT: util.optimization_barrier %[[RET]]
  util.optimization_barrier %0 : tensor<0x?xf32>
  return
}

// -----

// CHECK-LABEL: @reshapeEmpty
// CHECK-SAME: (%[[DIM:.+]]: index)
func.func @reshapeEmpty(%dim: index) -> tensor<?xi32> {
  // CHECK: %[[RET:.+]] = flow.tensor.empty : tensor<?xi32>{%[[DIM]]}
  %0 = flow.tensor.empty : tensor<1x?xi32>{%dim}
  // CHECK-NOT: flow.tensor.reshape
  %1 = flow.tensor.reshape %0 : tensor<1x?xi32>{%dim} -> tensor<?xi32>{%dim}
  // CHECK: return %[[RET]]
  return %1 : tensor<?xi32>
}

// -----

// CHECK-LABEL: @loadConst
func.func @loadConst() -> i32 {
  %0 = arith.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NEXT: %[[C2:.+]] = arith.constant 2 : i32
  %2 = flow.tensor.load %0[%c1, %c0] : tensor<2x2xi32>
  // CHECK-NEXT: return %[[C2]]
  return %2 : i32
}

// -----

// CHECK-LABEL: @loadConstScalar
func.func @loadConstScalar() -> i32 {
  %0 = arith.constant dense<4> : tensor<i32>
  // CHECK-NEXT: %[[C4:.+]] = arith.constant 4 : i32
  %1 = flow.tensor.load %0 : tensor<i32>
  // CHECK-NEXT: return %[[C4]]
  return %1 : i32
}

// -----

// CHECK-LABEL: @storeConst
func.func @storeConst() -> tensor<2x2xi32> {
  %0 = arith.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<[
  // CHECK-SAME:     [0, 1], [4, 3]
  // CHECK-SAME: ]> : tensor<2x2xi32>
  %1 = flow.tensor.store %c4, %0[%c1, %c0] : tensor<2x2xi32>
  // CHECK-NEXT: return %[[C]]
  return %1 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: @storeConstScalar
func.func @storeConstScalar() -> tensor<i32> {
  %0 = arith.constant dense<0> : tensor<i32>
  %1 = arith.constant 4 : i32
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<4> : tensor<i32>
  %2 = flow.tensor.store %1, %0 : tensor<i32>
  // CHECK-NEXT: return %[[C]]
  return %2 : tensor<i32>
}

// -----

// CHECK-LABEL: @allocaDims
//  CHECK-SAME: (%[[DIM:.+]]: index)
func.func @allocaDims(%dim: index) -> (index, index, index) {
  // CHECK-NOT: flow.tensor.alloca
  %0 = flow.tensor.alloca : tensor<4x?x0xf32>{%dim}
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %0, %c0 : tensor<4x?x0xf32>
  %d1 = tensor.dim %0, %c1 : tensor<4x?x0xf32>
  %d2 = tensor.dim %0, %c2 : tensor<4x?x0xf32>
  // CHECK: return %c4, %[[DIM]], %c0
  return %d0, %d1, %d2 : index, index, index
}

// -----

// CHECK-LABEL: @emptyDims
//  CHECK-SAME: (%[[DIM:.+]]: index)
func.func @emptyDims(%dim: index) -> (index, index, index) {
  // CHECK-NOT: flow.tensor.empty
  %0 = flow.tensor.empty : tensor<4x?x0xf32>{%dim}
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %0, %c0 : tensor<4x?x0xf32>
  %d1 = tensor.dim %0, %c1 : tensor<4x?x0xf32>
  %d2 = tensor.dim %0, %c2 : tensor<4x?x0xf32>
  // CHECK: return %c4, %[[DIM]], %c0
  return %d0, %d1, %d2 : index, index, index
}

// -----

// CHECK-LABEL: @splatDynamicShape
//  CHECK-SAME: (%[[DIM0:.+]]: index, %[[DIM1:.+]]: index)
func.func @splatDynamicShape(%dim0: index, %dim1: index) -> tensor<?x?xi32> {
  // CHECK: %[[FOUR:.+]] = arith.constant 4 : i32
  %four = arith.constant 4 : i32
  // CHECK: %[[SPLAT:.+]] = flow.tensor.splat %[[FOUR]] : tensor<?x?xi32>{%[[DIM0]], %[[DIM1]]}
  %1 = flow.tensor.splat %four : tensor<?x?xi32>{%dim0, %dim1}
  // CHECK: return %[[SPLAT]]
  return %1 : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: @splatStaticZeroElements
func.func @splatStaticZeroElements(%value: f32) -> tensor<0x2xf32> {
  // CHECK-NEXT: %[[RET:.+]] = flow.tensor.empty : tensor<0x2xf32>
  %0 = flow.tensor.splat %value : tensor<0x2xf32>
  // CHECK-NEXT: return %[[RET]]
  return %0 : tensor<0x2xf32>
}

// -----

// CHECK-LABEL: @splatDynamicZeroElements
//  CHECK-SAME: (%[[VALUE:.+]]: f32, %[[DIM:.+]]: index)
func.func @splatDynamicZeroElements(%value: f32, %dim: index) -> tensor<0x?xf32> {
  // CHECK-NEXT: %[[RET:.+]] = flow.tensor.empty : tensor<0x?xf32>{%[[DIM]]}
  %0 = flow.tensor.splat %value : tensor<0x?xf32>{%dim}
  // CHECK-NEXT: return %[[RET]]
  return %0 : tensor<0x?xf32>
}

// -----

// CHECK-LABEL: @cloneConst
func.func @cloneConst() -> tensor<4xi32> {
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %0 = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %1 = flow.tensor.clone %0 : tensor<4xi32>
  // CHECK-NEXT: return %[[C]]
  return %1 : tensor<4xi32>
}

// -----

// CHECK-LABEL: @cloneConstZeroElements
func.func @cloneConstZeroElements() -> tensor<0x2xi32> {
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<> : tensor<0x2xi32>
  %0 = arith.constant dense<> : tensor<0x2xi32>
  // CHECK-NOT: flow.tensor.clone
  %1 = flow.tensor.clone %0 : tensor<0x2xi32>
  // CHECK-NEXT: return %[[C]]
  return %1 : tensor<0x2xi32>
}

// -----

// CHECK-LABEL: @cloneStaticZeroElements
func.func @cloneStaticZeroElements(%arg0: tensor<0x2xf32>) -> tensor<0x2xf32> {
  // CHECK-NOT: flow.tensor.clone
  // CHECK-NEXT: %[[RET:.+]] = flow.tensor.empty : tensor<0x2xf32>
  %0 = flow.tensor.clone %arg0 : tensor<0x2xf32>
  // CHECK-NEXT: %[[RET]]
  return %0 : tensor<0x2xf32>
}

// -----

// CHECK-LABEL: @cloneDynamicZeroElements
//  CHECK-SAME: (%[[OPERAND:.+]]: tensor<0x?xf32>, %[[DIM:.+]]: index)
func.func @cloneDynamicZeroElements(%arg0: tensor<0x?xf32>, %dim: index) -> tensor<0x?xf32> {
  // CHECK-NOT: flow.tensor.clone
  // CHECK-NEXT: %[[RET:.+]] = flow.tensor.empty : tensor<0x?xf32>{%[[DIM]]}
  %0 = flow.tensor.clone %arg0 : tensor<0x?xf32>{%dim}
  // CHECK-NEXT: %[[RET]]
  return %0 : tensor<0x?xf32>
}

// -----

// CHECK-LABEL: @sliceConst0D
func.func @sliceConst0D() -> tensor<i32> {
  %0 = arith.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<0> : tensor<i32>
  %1 = flow.tensor.slice %0[for] : tensor<i32> -> tensor<i32>
  // CHECK-NEXT: return %[[C]]
  return %1 : tensor<i32>
}

// -----

// CHECK-LABEL: @sliceConst1D
func.func @sliceConst1D() -> tensor<1xi32> {
  %0 = arith.constant dense<0> : tensor<1xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<0> : tensor<1xi32>
  %1 = flow.tensor.slice %0[%c0 for %c1] : tensor<1xi32> -> tensor<1xi32>
  // CHECK-NEXT: return %[[C]]
  return %1 : tensor<1xi32>
}

// -----

// CHECK-LABEL: @sliceConst1DZeroLength
func.func @sliceConst1DZeroLength() -> tensor<0xi32> {
  %0 = arith.constant dense<0> : tensor<1xi32>
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<> : tensor<0xi32>
  %1 = flow.tensor.slice %0[%c0 for %c0] : tensor<1xi32> -> tensor<0xi32>
  // CHECK-NEXT: return %[[C]]
  return %1 : tensor<0xi32>
}

// -----

// CHECK-LABEL: @sliceConst2D
func.func @sliceConst2D() -> tensor<1x2xi32> {
  %0 = arith.constant dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<[
  // CHECK-SAME:     [1, 2]
  // CHECK-SAME: ]> : tensor<1x2xi32>
  %1 = flow.tensor.slice %0[%c0, %c1 for %c1, %c2] : tensor<2x3xi32> -> tensor<1x2xi32>
  // CHECK-NEXT: return %[[C]]
  return %1 : tensor<1x2xi32>
}

// -----

// CHECK-LABEL: @sliceConst2DZeroLength1
func.func @sliceConst2DZeroLength1() -> tensor<1x0xi32> {
  %0 = arith.constant dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<> : tensor<1x0xi32>
  %1 = flow.tensor.slice %0[%c0, %c0 for %c1, %c0] : tensor<2x3xi32> -> tensor<1x0xi32>
  // CHECK-NEXT: return %[[C]]
  return %1 : tensor<1x0xi32>
}

// -----

// CHECK-LABEL: @sliceConst2DZeroLength01
func.func @sliceConst2DZeroLength01() -> tensor<0x0xi32> {
  %0 = arith.constant dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi32>
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<> : tensor<0x0xi32>
  %1 = flow.tensor.slice %0[%c0, %c0 for %c0, %c0] : tensor<2x3xi32> -> tensor<0x0xi32>
  // CHECK-NEXT: return %[[C]]
  return %1 : tensor<0x0xi32>
}

// -----

// CHECK-LABEL: @sliceFromZeroElements
func.func @sliceFromZeroElements(%arg0: tensor<0xi32>) -> tensor<?xi32> {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: flow.tensor.slice
  // CHECK: %[[RET:.+]] = flow.tensor.empty : tensor<?xi32>{%c0}
  %0 = flow.tensor.slice %arg0[%c0 for %c0] : tensor<0xi32> -> tensor<?xi32>{%c0}
  // CHECK: return %[[RET]]
  return %0 : tensor<?xi32>
}

// -----

// CHECK-LABEL: @sliceZeroElements
func.func @sliceZeroElements(%arg0: tensor<?xi32>, %dim: index) -> tensor<0xi32> {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: flow.tensor.slice
  // CHECK: %[[RET:.+]] = flow.tensor.empty : tensor<0xi32>
  %0 = flow.tensor.slice %arg0[%c0 for %c0] : tensor<?xi32>{%dim} -> tensor<0xi32>
  // CHECK: return %[[RET]]
  return %0 : tensor<0xi32>
}

// -----

// CHECK-LABEL: @sliceConst3D
func.func @sliceConst3D() -> tensor<1x2x3xi32> {
  %0 = arith.constant dense<[[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[9, 10, 11], [12, 13, 14], [15, 16, 17]]]> : tensor<2x3x3xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<[
  // CHECK-SAME:                             [
  // CHECK-SAME:                              [3, 4, 5], [6, 7, 8]]]> : tensor<1x2x3xi32>
  %1 = flow.tensor.slice %0[%c0, %c1, %c0 for %c1, %c2, %c3] : tensor<2x3x3xi32> -> tensor<1x2x3xi32>
  // CHECK-NEXT: return %[[C]]
  return %1 : tensor<1x2x3xi32>
}

// -----

// CHECK-LABEL: @updateConst0D
func.func @updateConst0D() -> tensor<i32> {
  %0 = arith.constant dense<0> : tensor<i32>
  %1 = arith.constant dense<1> : tensor<i32>
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<0> : tensor<i32>
  %2 = flow.tensor.update %0, %1[] : tensor<i32> -> tensor<i32>
  // CHECK-NEXT: return %[[C]]
  return %2 : tensor<i32>
}

// -----

// CHECK-LABEL: @updateConst1D
func.func @updateConst1D() -> tensor<1xi32> {
  %0 = arith.constant dense<0> : tensor<1xi32>
  %1 = arith.constant dense<1> : tensor<1xi32>
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<0> : tensor<1xi32>
  %2 = flow.tensor.update %0, %1[%c0] : tensor<1xi32> -> tensor<1xi32>
  // CHECK-NEXT: return %[[C]]
  return %2 : tensor<1xi32>
}

// -----

// CHECK-LABEL: @updateConst1DUpdateZeroSize
func.func @updateConst1DUpdateZeroSize() -> tensor<1xi32> {
  %0 = arith.constant dense<> : tensor<0xi32>
  %1 = arith.constant dense<1> : tensor<1xi32>
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<1> : tensor<1xi32>
  %2 = flow.tensor.update %0, %1[%c0] : tensor<0xi32> -> tensor<1xi32>
  // CHECK-NEXT: return %[[C]]
  return %2 : tensor<1xi32>
}

// -----

// CHECK-LABEL: @updateConst2DUpdate1x1
func.func @updateConst2DUpdate1x1() -> tensor<3x4xi32> {
  %0 = arith.constant dense<[[12]]> : tensor<1x1xi32>
  %1 = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]> : tensor<3x4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<[
  // CHECK-SAME: [0, 12, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]> : tensor<3x4xi32>
  %2 = flow.tensor.update %0, %1[%c0, %c1] : tensor<1x1xi32> -> tensor<3x4xi32>
  // CHECK-NEXT: return %[[C]]
  return %2 : tensor<3x4xi32>
}

// -----

// CHECK-LABEL: @updateConst2DUpdate2x2
func.func @updateConst2DUpdate2x2() -> tensor<3x4xi32> {
  %0 = arith.constant dense<[[12, 13], [14, 15]]> : tensor<2x2xi32>
  %1 = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]> : tensor<3x4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<[
  // CHECK-SAME: [0, 12, 13, 3], [4, 14, 15, 7], [8, 9, 10, 11]]> : tensor<3x4xi32>
  %2 = flow.tensor.update %0, %1[%c0, %c1] : tensor<2x2xi32> -> tensor<3x4xi32>
  // CHECK-NEXT: return %[[C]]
  return %2 : tensor<3x4xi32>
}

// -----

// CHECK-LABEL: @updateConst3DUpdate1x2x3
func.func @updateConst3DUpdate1x2x3() -> tensor<2x3x3xi32> {
  %0 = arith.constant dense<[[[18, 19, 20], [21, 22, 23]]]> : tensor<1x2x3xi32>
  %1 = arith.constant dense<[[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[9, 10, 11], [12, 13, 14], [15, 16, 17]]]> : tensor<2x3x3xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<[
  // CHECK-SAME:                             [
  // CHECK-SAME:                              [0, 1, 2], [18, 19, 20], [21, 22, 23]], [
  // CHECK-SAME: [9, 10, 11], [12, 13, 14], [15, 16, 17]]]> : tensor<2x3x3xi32>
  %2 = flow.tensor.update %0, %1[%c0, %c1, %c0] : tensor<1x2x3xi32> -> tensor<2x3x3xi32>
  // CHECK-NEXT: return %[[C]]
  return %2 : tensor<2x3x3xi32>
}

// -----

// CHECK-LABEL: @updateConst3DUpdate2x3x2
func.func @updateConst3DUpdate2x3x2() -> tensor<2x3x3xi32> {
  %0 = arith.constant dense<[[[18, 19], [20, 21], [22, 23]], [[24, 25], [26, 27], [28, 29]]]> : tensor<2x3x2xi32>
  %1 = arith.constant dense<[[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[9, 10, 11], [12, 13, 14], [15, 16, 17]]]> : tensor<2x3x3xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 0 : index
  // CHECK-NEXT: %[[C:.+]] = arith.constant dense<[
  // CHECK-SAME:                             [
  // CHECK-SAME:                              [18, 19, 2], [20, 21, 5], [22, 23, 8]], [
  // CHECK-SAME: [24, 25, 11], [26, 27, 14], [28, 29, 17]]]> : tensor<2x3x3xi32>
  %2 = flow.tensor.update %0, %1[%c0, %c1, %c0] : tensor<2x3x2xi32> -> tensor<2x3x3xi32>
  // CHECK-NEXT: return %[[C]]
  return %2 : tensor<2x3x3xi32>
}

// -----

// CHECK-LABEL: @updateReplace
func.func @updateReplace(%arg0 : tensor<4xi32>, %arg1 : tensor<4xi32>) -> tensor<4xi32> {
  %c0 = arith.constant 0 : index
  %0 = flow.tensor.update %arg0, %arg1[%c0] : tensor<4xi32> -> tensor<4xi32>
  // CHECK-NEXT: return %arg0
  return %0 : tensor<4xi32>
}

// -----

// CHECK-LABEL: @updateIntoZeroElements
func.func @updateIntoZeroElements(%update: tensor<?x?xi32>, %dim: index, %target: tensor<0x0xi32>) -> tensor<0x0xi32> {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: flow.tensor.update
  // CHECK-NEXT: %[[RET:.+]] = flow.tensor.empty : tensor<0x0xi32>
  %0 = flow.tensor.update %update, %target[%c0, %c0] : tensor<?x?xi32>{%dim, %dim} -> tensor<0x0xi32>
  // CHECK-NEXT: return %[[RET]]
  return %0 : tensor<0x0xi32>
}

// -----

// CHECK-LABEL: @updateZeroElements
//  CHECK-SAME: (%[[UPDATE:.+]]: tensor<0x1xi32>, %[[TARGET:.+]]: tensor<1x1xi32>)
func.func @updateZeroElements(%update: tensor<0x1xi32>, %target: tensor<1x1xi32>) -> tensor<1x1xi32> {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: flow.tensor.update
  %0 = flow.tensor.update %update, %target[%c0, %c0] : tensor<0x1xi32> -> tensor<1x1xi32>
  // CHECK: return %[[TARGET]]
  return %0 : tensor<1x1xi32>
}

// -----

// CHECK-LABEL: @propogateStaticShapeOfTarget
func.func @propogateStaticShapeOfTarget(%arg0 : tensor<?x?xf32>, %arg1 : f32) -> tensor<?x?xf32> {
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
  // CHECK: return %[[RESULT]]
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @propogateStaticShapeOfUpdate
func.func @propogateStaticShapeOfUpdate(%arg0 : tensor<?x?xf32>, %arg1 : f32) -> tensor<?x?xf32> {
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
  // CHECK: return %[[RESULT]]
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @foldSplatLoadIntoPrimitive
//  CHECK-SAME: (%[[arg0:.+]]: f32, %[[arg1:.+]]: index, %[[arg2:.+]]: index)
func.func @foldSplatLoadIntoPrimitive(%arg0 : f32, %arg1 : index, %arg2 : index) -> f32 {
  // CHECK-NEXT: return %[[arg0]] : f32
  %0 = flow.tensor.splat %arg0 : tensor<4x4xf32>
  %1 = flow.tensor.load %0[%arg1, %arg2] : tensor<4x4xf32>
  return %1 : f32
}

// -----

// CHECK-LABEL: @foldSplatReshapeIntoSplat
func.func @foldSplatReshapeIntoSplat(%arg0 : f32) -> tensor<16xf32> {
  // CHECK-NEXT: %0 = flow.tensor.splat %arg0 : tensor<16xf32>
  // CHECK-NEXT: return %0 : tensor<16xf32>
  %0 = flow.tensor.splat %arg0 : tensor<4x4xf32>
  %1 = flow.tensor.reshape %0 : tensor<4x4xf32> -> tensor<16xf32>
  return %1 : tensor<16xf32>
}

// -----

// CHECK-LABEL: @foldSplatReshapeIntoSplatDynamic
func.func @foldSplatReshapeIntoSplatDynamic(%arg0 : f32, %arg1 : index, %arg2 : index, %arg3 : index) -> tensor<?x?xf32> {
  // CHECK-NEXT: %0 = flow.tensor.splat %arg0 : tensor<?x?xf32>{%arg2, %arg3}
  // CHECK-NEXT: return %0 : tensor<?x?xf32>
  %0 = flow.tensor.splat %arg0 : tensor<?x4xf32>{%arg1}
  %1 = flow.tensor.reshape %0 : tensor<?x4xf32>{%arg1} -> tensor<?x?xf32>{%arg2, %arg3}
  return %1 : tensor<?x?xf32>
}

// -----

func.func @innermost_unit_dim(%4: !flow.dispatch.tensor<readonly:tensor<3x1x16x257x88xf16>>,
    %arg0: index, %arg2 : index, %10 : index, %9 : index) -> tensor<?x?x?xf16> {
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %11 = flow.dispatch.tensor.load %4, offsets = [1, 0, %arg0, %10, %arg2], sizes = [1, 1, %c16, %9, %c1], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x1x16x257x88xf16>> -> tensor<?x?x?xf16>
  return %11 : tensor<?x?x?xf16>
}
// CHECK-LABEL: func @innermost_unit_dim
//  CHECK-SAME:     %[[DYNAMIC_DIM:[a-zA-Z0-9]+]]: index)
//       CHECK:   flow.dispatch.tensor.load
//  CHECK-SAME:       sizes = [1, 1, 16, %[[DYNAMIC_DIM]], 1]

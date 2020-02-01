// RUN: iree-opt -split-input-file -verify-diagnostics -canonicalize %s | IreeFileCheck %s


// CHECK-LABEL: @elideTiedGetRankedShape
// CHECK-SAME: %[[T:[^:[:space:]]+]]: tensor<1x?x2x?xf32>
// CHECK-SAME: %[[SHAPE:[^:[:space:]]+]]: !shapex.ranked_shape<[1,?,2,?],i32>
func @elideTiedGetRankedShape(%arg0: tensor<1x?x2x?xf32>, %arg1: !shapex.ranked_shape<[1,?,2,?],i32>) -> (tensor<1x?x2x?xf32>, !shapex.ranked_shape<[1,?,2,?],i32>) {
  // Note that canonicalization does *not* remove tie_shape. That must be
  // removed manually once all shape materialization is complete (otherwise,
  // information needed to materialize would be lost).
  // CHECK: %[[TIE_T:.+]] = shapex.tie_shape %[[T]], %[[SHAPE]]
  %0 = shapex.tie_shape %arg0, %arg1 : tensor<1x?x2x?xf32>, !shapex.ranked_shape<[1,?,2,?],i32>
  // CHECK-NOT: shapex.get_ranked_shape
  %1 = shapex.get_ranked_shape %0 : tensor<1x?x2x?xf32> -> !shapex.ranked_shape<[1,?,2,?],i32>
  // CHECK-DAG: return %[[TIE_T]], %[[SHAPE]]
  return %0, %1 : tensor<1x?x2x?xf32>, !shapex.ranked_shape<[1,?,2,?],i32>
}

// -----
// CHECK-LABEL: @staticGetRankedShapeToConst
// CHECK-SAME: %[[T:[^:[:space:]]+]]: tensor<1x2xf32>
func @staticGetRankedShapeToConst(%arg0: tensor<1x2xf32>) -> (!shapex.ranked_shape<[1,2],i32>) {
  // CHECK-NOT: %[[T]]
  // CHECK: %[[S:.+]] = shapex.const_ranked_shape {value} : !shapex.ranked_shape<[1,2],i32>
  %0 = shapex.get_ranked_shape %arg0 : tensor<1x2xf32> -> !shapex.ranked_shape<[1,2],i32>
  // CHECK: return %[[S]]
  return %0 : !shapex.ranked_shape<[1,2],i32>
}

// -----
// CHECK-LABEL: @foldStaticRankedDim
// CHECK-SAME: %[[SHAPE:[^:[:space:]]+]]: !shapex.ranked_shape<[1,?,2,?],i32>
func @foldStaticRankedDim(%arg0: !shapex.ranked_shape<[1,?,2,?],i32>) -> (i32, i32) {
  // CHECK-DAG: %[[D2:.+]] = constant 2 : i32
  %0 = shapex.ranked_dim %arg0[2] : !shapex.ranked_shape<[1,?,2,?],i32>
  // CHECK-DAG: %[[D1:.+]] = shapex.ranked_dim %[[SHAPE]][1]
  %1 = shapex.ranked_dim %arg0[1] : !shapex.ranked_shape<[1,?,2,?],i32>
  // CHECK: return %[[D2]], %[[D1]]
  return %0, %1 : i32, i32
}

// -----
// CHECK-LABEL: @foldFullyStaticRankedShape
// CHECK-SAME: %[[T:[^:[:space:]]+]]: tensor<1x2xf32>
func @foldFullyStaticRankedShape(%arg0: tensor<1x2xf32>) -> (i32, i32) {
  // CHECK-NOT: shapex.get_ranked_shape
  // CHECK-NOT: shapex.ranked_dim
  // CHECK: constant 1
  // CHECK: constant 2
  %0 = shapex.get_ranked_shape %arg0 : tensor<1x2xf32> -> !shapex.ranked_shape<[1,2],i32>
  %1 = shapex.ranked_dim %0[0] : !shapex.ranked_shape<[1,2],i32>
  %2 = shapex.ranked_dim %0[1] : !shapex.ranked_shape<[1,2],i32>
  return %1, %2 : i32, i32
}

// -----
// CHECK-LABEL: @dynamicMakeRankedShapeDim
// CHECK-SAME: %[[DD0:[^:[:space:]]+]]: index
// CHECK-SAME: %[[DD1:[^:[:space:]]+]]: index
func @dynamicMakeRankedShapeDim(%arg0: index, %arg1 : index) -> (index, index, index, index) {
  // CHECK-NOT: make_ranked_shape
  // CHECK-NOT: ranked_dim
  %rs = shapex.make_ranked_shape %arg0, %arg1 -> !shapex.ranked_shape<[?,8,?,16]>
  %d0 = shapex.ranked_dim %rs[0] : !shapex.ranked_shape<[?,8,?,16]>
  %d1 = shapex.ranked_dim %rs[1] : !shapex.ranked_shape<[?,8,?,16]>
  %d2 = shapex.ranked_dim %rs[2] : !shapex.ranked_shape<[?,8,?,16]>
  %d3 = shapex.ranked_dim %rs[3] : !shapex.ranked_shape<[?,8,?,16]>
  // CHECK-DAG: %[[C8:.+]] = constant 8 : index
  // CHECK-DAG: %[[C16:.+]] = constant 16 : index
  // CHECK-DAG: return %[[DD0]], %[[C8]], %[[DD1]], %[[C16]]
  return %d0, %d1, %d2, %d3 : index, index, index, index
}

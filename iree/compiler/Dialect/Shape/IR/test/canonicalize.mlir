// RUN: iree-opt -split-input-file -verify-diagnostics -canonicalize %s | IreeFileCheck %s


// CHECK-LABEL: @elideTiedGetRankedShape
// CHECK-SAME: %[[T:[^:[:space:]]+]]: tensor<1x?x2x?xf32>
// CHECK-SAME: %[[SHAPE:[^:[:space:]]+]]: !shape.ranked_shape<1x?x2x?xi32>
func @elideTiedGetRankedShape(%arg0: tensor<1x?x2x?xf32>, %arg1: !shape.ranked_shape<1x?x2x?xi32>) -> (tensor<1x?x2x?xf32>, !shape.ranked_shape<1x?x2x?xi32>) {
  // Note that canonicalization does *not* remove tie_shape. That must be
  // removed manually once all shape materialization is complete (otherwise,
  // information needed to materialize would be lost).
  // CHECK: %[[TIE_T:.+]] = shape.tie_shape %[[T]], %[[SHAPE]]
  %0 = shape.tie_shape %arg0, %arg1 : tensor<1x?x2x?xf32>, !shape.ranked_shape<1x?x2x?xi32>
  // CHECK-NOT: shape.get_ranked_shape
  %1 = shape.get_ranked_shape %0 : tensor<1x?x2x?xf32> -> !shape.ranked_shape<1x?x2x?xi32>
  // CHECK-DAG: return %[[TIE_T]], %[[SHAPE]]
  return %0, %1 : tensor<1x?x2x?xf32>, !shape.ranked_shape<1x?x2x?xi32>
}

// -----
// CHECK-LABEL: @staticGetRankedShapeToConst
// CHECK-SAME: %[[T:[^:[:space:]]+]]: tensor<1x2xf32>
func @staticGetRankedShapeToConst(%arg0: tensor<1x2xf32>) -> (!shape.ranked_shape<1x2xi32>) {
  // CHECK-NOT: %[[T]]
  // CHECK: %[[S:.+]] = shape.const_ranked_shape : !shape.ranked_shape<1x2xi32>
  %0 = shape.get_ranked_shape %arg0 : tensor<1x2xf32> -> !shape.ranked_shape<1x2xi32>
  // CHECK: return %[[S]]
  return %0 : !shape.ranked_shape<1x2xi32>
}

// -----
// CHECK-LABEL: @foldStaticRankedDim
// CHECK-SAME: %[[SHAPE:[^:[:space:]]+]]: !shape.ranked_shape<1x?x2x?xi32>
func @foldStaticRankedDim(%arg0: !shape.ranked_shape<1x?x2x?xi32>) -> (i32, i32) {
  // CHECK-DAG: %[[D2:.+]] = constant 2 : i32
  %0 = shape.ranked_dim %arg0[2] : !shape.ranked_shape<1x?x2x?xi32>
  // CHECK-DAG: %[[D1:.+]] = shape.ranked_dim %[[SHAPE]][1]
  %1 = shape.ranked_dim %arg0[1] : !shape.ranked_shape<1x?x2x?xi32>
  // CHECK: return %[[D2]], %[[D1]]
  return %0, %1 : i32, i32
}

// -----
// CHECK-LABEL: @foldFullyStaticRankedShape
// CHECK-SAME: %[[T:[^:[:space:]]+]]: tensor<1x2xf32>
func @foldFullyStaticRankedShape(%arg0: tensor<1x2xf32>) -> (i32, i32) {
  // CHECK-NOT: shape.get_ranked_shape
  // CHECK-NOT: shape.ranked_dim
  // CHECK: constant 1
  // CHECK: constant 2
  %0 = shape.get_ranked_shape %arg0 : tensor<1x2xf32> -> !shape.ranked_shape<1x2xi32>
  %1 = shape.ranked_dim %0[0] : !shape.ranked_shape<1x2xi32>
  %2 = shape.ranked_dim %0[1] : !shape.ranked_shape<1x2xi32>
  return %1, %2 : i32, i32
}

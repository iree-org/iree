// RUN: iree-opt -split-input-file -verify-diagnostics -canonicalize %s | IreeFileCheck %s

// -----
// CHECK-LABEL: @foldStaticRankedDim
// CHECK-SAME: %[[SHAPE:[^:[:space:]]+]]: !shapex.ranked_shape<[1,?,2,?]>
func @foldStaticRankedDim(%arg0: !shapex.ranked_shape<[1,?,2,?]>) -> (i32, i32) {
  // CHECK-DAG: %[[D2:.+]] = arith.constant 2 : i32
  %0 = shapex.ranked_dim %arg0[2] : !shapex.ranked_shape<[1,?,2,?]> -> i32
  // CHECK-DAG: %[[D1:.+]] = shapex.ranked_dim %[[SHAPE]][1]
  %1 = shapex.ranked_dim %arg0[1] : !shapex.ranked_shape<[1,?,2,?]> -> i32
  // CHECK: return %[[D2]], %[[D1]]
  return %0, %1 : i32, i32
}

// -----
// CHECK-LABEL: @dynamicMakeRankedShapeDim
// CHECK-SAME: %[[DD0:[^:[:space:]]+]]: index
// CHECK-SAME: %[[DD1:[^:[:space:]]+]]: index
func @dynamicMakeRankedShapeDim(%arg0: index, %arg1 : index) -> (index, index, index, index) {
  // CHECK-NOT: make_ranked_shape
  // CHECK-NOT: ranked_dim
  %rs = shapex.make_ranked_shape %arg0, %arg1 : (index, index) -> !shapex.ranked_shape<[?,8,?,16]>
  %d0 = shapex.ranked_dim %rs[0] : !shapex.ranked_shape<[?,8,?,16]> -> index
  %d1 = shapex.ranked_dim %rs[1] : !shapex.ranked_shape<[?,8,?,16]> -> index
  %d2 = shapex.ranked_dim %rs[2] : !shapex.ranked_shape<[?,8,?,16]> -> index
  %d3 = shapex.ranked_dim %rs[3] : !shapex.ranked_shape<[?,8,?,16]> -> index
  // CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
  // CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
  // CHECK-DAG: return %[[DD0]], %[[C8]], %[[DD1]], %[[C16]]
  return %d0, %d1, %d2, %d3 : index, index, index, index
}

//===----------------------------------------------------------------------===//
// IdentityMakeRankedShapePattern tests
//===----------------------------------------------------------------------===//

// -----
// CHECK-LABEL: @identityMakeRankedShape_match_1dim
// CHECK-SAME: %[[ARGRS:[^:[:space:]]+]]: !shapex.ranked_shape
func @identityMakeRankedShape_match_1dim(%arg0 : !shapex.ranked_shape<[?,16]>) -> !shapex.ranked_shape<[?,16]> {
  // CHECK-NOT: shapex.make_ranked_shape
  %0 = shapex.ranked_dim %arg0[0] : !shapex.ranked_shape<[?,16]> -> index
  %1 = shapex.make_ranked_shape %0 : (index) -> !shapex.ranked_shape<[?,16]>
  // CHECK: return %[[ARGRS]]
  return %1 : !shapex.ranked_shape<[?,16]>
}

// -----
// CHECK-LABEL: @identityMakeRankedShape_match_2dim
// CHECK-SAME: %[[ARGRS:[^:[:space:]]+]]: !shapex.ranked_shape
func @identityMakeRankedShape_match_2dim(%arg0 : !shapex.ranked_shape<[?,16,?]>) -> !shapex.ranked_shape<[?,16,?]> {
  // CHECK-NOT: shapex.make_ranked_shape
  %0 = shapex.ranked_dim %arg0[0] : !shapex.ranked_shape<[?,16,?]> -> index
  %1 = shapex.ranked_dim %arg0[2] : !shapex.ranked_shape<[?,16,?]> -> index
  %2 = shapex.make_ranked_shape %0, %1 : (index, index) -> !shapex.ranked_shape<[?,16,?]>
  // CHECK: return %[[ARGRS]]
  return %2 : !shapex.ranked_shape<[?,16,?]>
}

// -----
// CHECK-LABEL: @identityMakeRankedShape_nomatch_swap_dims
// CHECK-SAME: %[[ARGRS:[^:[:space:]]+]]: !shapex.ranked_shape
func @identityMakeRankedShape_nomatch_swap_dims(%arg0 : !shapex.ranked_shape<[?,16,?]>) -> !shapex.ranked_shape<[?,16,?]> {
  %0 = shapex.ranked_dim %arg0[2] : !shapex.ranked_shape<[?,16,?]> -> index
  %1 = shapex.ranked_dim %arg0[0] : !shapex.ranked_shape<[?,16,?]> -> index
  %2 = shapex.make_ranked_shape %0, %1 : (index, index) -> !shapex.ranked_shape<[?,16,?]>
  // CHECK: %[[RS:.+]] = shapex.make_ranked_shape
  // CHECK: return %[[RS]]
  return %2 : !shapex.ranked_shape<[?,16,?]>
}

// -----
// CHECK-LABEL: @identityMakeRankedShape_nomatch_static_dim
// CHECK-SAME: %[[ARGRS:[^:[:space:]]+]]: !shapex.ranked_shape
func @identityMakeRankedShape_nomatch_static_dim(%arg0 : !shapex.ranked_shape<[?,16,?]>) -> !shapex.ranked_shape<[?,16,?]> {
  %0 = shapex.ranked_dim %arg0[1] : !shapex.ranked_shape<[?,16,?]> -> index
  %1 = shapex.ranked_dim %arg0[2] : !shapex.ranked_shape<[?,16,?]> -> index
  %2 = shapex.make_ranked_shape %0, %1 : (index, index) -> !shapex.ranked_shape<[?,16,?]>
  // CHECK: %[[RS:.+]] = shapex.make_ranked_shape
  // CHECK: return %[[RS]]
  return %2 : !shapex.ranked_shape<[?,16,?]>
}

// CHECK-LABEL: @identityMakeRankedShape_nomatch_different_shape
// CHECK-SAME: %[[ARGRS1:[^:[:space:]]+]]: !shapex.ranked_shape
// CHECK-SAME: %[[ARGRS2:[^:[:space:]]+]]: !shapex.ranked_shape
func @identityMakeRankedShape_nomatch_different_shape(%arg0 : !shapex.ranked_shape<[?,16,?]>, %arg1 : !shapex.ranked_shape<[?,16,?]>) -> !shapex.ranked_shape<[?,16,?]> {
  %0 = shapex.ranked_dim %arg0[0] : !shapex.ranked_shape<[?,16,?]> -> index
  %1 = shapex.ranked_dim %arg1[2] : !shapex.ranked_shape<[?,16,?]> -> index
  %2 = shapex.make_ranked_shape %0, %1 : (index, index) -> !shapex.ranked_shape<[?,16,?]>
  // CHECK: %[[RS:.+]] = shapex.make_ranked_shape
  // CHECK: return %[[RS]]
  return %2 : !shapex.ranked_shape<[?,16,?]>
}

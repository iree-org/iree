// RUN: iree-opt -split-input-file -verify-diagnostics -iree-shape-expand-function-ranked-shape-dims %s | IreeFileCheck %s

// CHECK-LABEL: @noOp
// CHECK-NOT: index
func @noOp(%arg0 : tensor<1x2xf32>) {
  return
}

// -----
// CHECK-LABEL: @expandInputs
// CHECK-SAME: %[[T:[^:[:space:]]+]]: tensor<?x2xf32>
// CHECK-SAME: %[[IDX0:[^:[:space:]]+]]: index
// CHECK-SAME: %[[IDX1:[^:[:space:]]+]]: index
func @expandInputs(%arg0 : tensor<?x2xf32>, %arg1 : !shapex.ranked_shape<[?,?]>, %arg3 : tensor<1xf32>) -> (index, index) {
  // CHECK-DAG: %[[RS:.+]] = shapex.make_ranked_shape %[[IDX0]], %[[IDX1]] : (index, index) -> !shapex.ranked_shape<[?,?]>
  // CHECK-DAG: %[[CAST_IDX0:.+]] = shapex.ranked_dim %[[RS]][0]
  // CHECK-DAG: %[[CAST_IDX1:.+]] = shapex.ranked_dim %[[RS]][1]
  %0 = shapex.ranked_dim %arg1[0] : !shapex.ranked_shape<[?,?]> -> index
  %1 = shapex.ranked_dim %arg1[1] : !shapex.ranked_shape<[?,?]> -> index
  // CHECK: return %[[CAST_IDX0]], %[[CAST_IDX1]]
  return %0, %1 : index, index
}

// -----
// CHECK-LABEL: @expandResults
// CHECK-SAME: () -> (index, index)
func @expandResults() -> (!shapex.ranked_shape<[?,?]>) {
  %idx0 = arith.constant 5 : index
  %idx1 = arith.constant 6 : index
  // CHECK: %[[RS:.+]] = shapex.make_ranked_shape
  %rs = shapex.make_ranked_shape %idx0, %idx1 : (index, index) -> !shapex.ranked_shape<[?,?]>
  // CHECK-DAG: %[[CAST_IDX0:.+]] = shapex.ranked_dim %[[RS]][0]
  // CHECK-DAG: %[[CAST_IDX1:.+]] = shapex.ranked_dim %[[RS]][1]
  // CHECK: return %[[CAST_IDX0]], %[[CAST_IDX1]]
  return %rs : !shapex.ranked_shape<[?,?]>
}

// -----
// CHECK-LABEL: @calls
// CHECK-SAME: (%arg0: index, %arg1: index) -> (index, index)
func @calls(%arg0 :!shapex.ranked_shape<[?,?]>) -> !shapex.ranked_shape<[?,?]> {
  // CHECK: call @calls(%{{.*}}, %{{.*}}) : (index, index) -> (index, index)
  %0 = std.call @calls(%arg0) : (!shapex.ranked_shape<[?,?]>) -> !shapex.ranked_shape<[?,?]>
  return %0 : !shapex.ranked_shape<[?,?]>
}

// -----
// CHECK-LABEL:   func @oneUnknownDimension(
// CHECK-SAME:                              %[[ARG:.*]]: index) -> index {
// CHECK:           %[[ARG_RS:.*]] = shapex.make_ranked_shape %[[ARG]] : (index) -> !shapex.ranked_shape<[?]>
// CHECK:           %[[ARG_DIM0:.*]] = shapex.ranked_dim %[[ARG_RS]][0] : !shapex.ranked_shape<[?]> -> index
// CHECK:           %[[CALL:.*]] = call @oneUnknownDimension(%[[ARG_DIM0]]) : (index) -> index
// CHECK:           %[[CALL_SHAPE:.*]] = shapex.make_ranked_shape %[[CALL]] : (index) -> !shapex.ranked_shape<[?]>
// CHECK:           %[[CALL_DIM0:.*]] = shapex.ranked_dim %[[CALL_SHAPE]][0] : !shapex.ranked_shape<[?]> -> index
// CHECK:           return %[[CALL_DIM0]] : index

func @oneUnknownDimension(%arg0 :!shapex.ranked_shape<[?]>) -> !shapex.ranked_shape<[?]> {
  %0 = std.call @oneUnknownDimension(%arg0) : (!shapex.ranked_shape<[?]>) -> !shapex.ranked_shape<[?]>
  return %0 : !shapex.ranked_shape<[?]>
}

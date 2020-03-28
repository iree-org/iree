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
  %idx0 = constant 5 : index
  %idx1 = constant 6 : index
  // CHECK: %[[RS:.+]] = shapex.make_ranked_shape
  %rs = shapex.make_ranked_shape %idx0, %idx1 : (index, index) -> !shapex.ranked_shape<[?,?]>
  // CHECK-DAG: %[[CAST_IDX0:.+]] = shapex.ranked_dim %[[RS]][0]
  // CHECK-DAG: %[[CAST_IDX1:.+]] = shapex.ranked_dim %[[RS]][1]
  // CHECK: return %[[CAST_IDX0]], %[[CAST_IDX1]]
  return %rs : !shapex.ranked_shape<[?,?]>
}

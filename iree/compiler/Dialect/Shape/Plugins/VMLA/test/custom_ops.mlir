// RUN: iree-opt -split-input-file -verify-diagnostics -iree-shape-materialize-calculations %s | IreeFileCheck %s

// -----
// CHECK-LABEL: func @batch.matmul.pseudo
func @batch.matmul.pseudo(
  %lhs: tensor<?x?x?xf32>, %rhs: tensor<?x?x?xf32>,
  %lhsShape: !shapex.ranked_shape<[?,?,?]>, %rhsShape: !shapex.ranked_shape<[?,?,?]>
) -> !shapex.ranked_shape<[?,?,?]> {
  %lhsTied = shapex.tie_shape %lhs, %lhsShape : tensor<?x?x?xf32>, !shapex.ranked_shape<[?,?,?]>
  %rhsTied = shapex.tie_shape %rhs, %rhsShape : tensor<?x?x?xf32>, !shapex.ranked_shape<[?,?,?]>
  %0 = "vmla.batch.matmul.pseudo"(%lhsTied, %rhsTied) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // CHECK-DAG: %[[BATCH:.+]] = shapex.ranked_dim %arg2[0]
  // CHECK-DAG: %[[FLHS:.+]] = shapex.ranked_dim %arg2[1]
  // CHECK-DAG: %[[FRHS:.+]] = shapex.ranked_dim %arg3[1]
  // CHECK-DAG: %[[SHAPE:.+]] = shapex.make_ranked_shape %[[BATCH]], %[[FRHS]], %[[FLHS]]
  // CHECK-DAG: return %[[SHAPE]]
  %1 = shapex.get_ranked_shape %0 : tensor<?x?x?xf32> -> !shapex.ranked_shape<[?,?,?]>
  return %1 : !shapex.ranked_shape<[?,?,?]>
}

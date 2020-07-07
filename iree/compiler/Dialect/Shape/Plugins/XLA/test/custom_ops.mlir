// RUN: iree-opt -split-input-file -verify-diagnostics -iree-shape-materialize-calculations %s | IreeFileCheck %s

// CHECK-LABEL: @transpose
func @transpose(%arg0: tensor<?x7x10xf32>, %arg1: !shapex.ranked_shape<[?,7,10]>) -> (tensor<7x?x10xf32>, !shapex.ranked_shape<[7,?,10]>) {
  %tied = shapex.tie_shape %arg0, %arg1 : tensor<?x7x10xf32>, !shapex.ranked_shape<[?,7,10]>
  %0 = "mhlo.transpose"(%tied) {permutation = dense<[1, 0, 2]> : tensor<3xi64>} :
      (tensor<?x7x10xf32>) -> tensor<7x?x10xf32>
  // CHECK-DAG: %[[RESULT:.+]] = "mhlo.transpose"
  // CHECK-DAG: %[[DIM:.+]] = shapex.ranked_dim %arg1[0]
  // CHECK-DAG: %[[SHAPE:.+]] = shapex.make_ranked_shape %[[DIM]]
  %1 = shapex.get_ranked_shape %0 : tensor<7x?x10xf32> -> !shapex.ranked_shape<[7,?,10]>
  // CHECK: return %[[RESULT]], %[[SHAPE]]
  return %0, %1 : tensor<7x?x10xf32>, !shapex.ranked_shape<[7,?,10]>
}

// -----
// CHECK-LABEL: func @dot_general
func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>,
                  %arg2: !shapex.ranked_shape<[?,?,?]>, %arg3: !shapex.ranked_shape<[?,?,?]>) -> !shapex.ranked_shape<[?,?,?]> {
  %tie0 = shapex.tie_shape %arg0, %arg2 : tensor<?x?x?xf32>, !shapex.ranked_shape<[?,?,?]>
  %tie1 = shapex.tie_shape %arg1, %arg3 : tensor<?x?x?xf32>, !shapex.ranked_shape<[?,?,?]>

  // Extents are taken directly from args.
  // CHECK-DAG: %[[EXTENT0:.+]] = shapex.ranked_dim %arg2[0]
  // CHECK-DAG: %[[EXTENT1:.+]] = shapex.ranked_dim %arg2[1]
  // CHECK-DAG: %[[EXTENT2:.+]] = shapex.ranked_dim %arg3[2]
  // CHECK-DAG: %[[SHAPE:.+]] = shapex.make_ranked_shape %[[EXTENT0]], %[[EXTENT1]], %[[EXTENT2]]
  // CHECK-DAG: return %[[SHAPE]]
  %0 = "mhlo.dot_general"(%tie0, %tie1) { dot_dimension_numbers = {
    lhs_batching_dimensions = dense<0> : tensor<1xi64>,
    lhs_contracting_dimensions = dense<2> : tensor<1xi64>,
    rhs_batching_dimensions = dense<0> : tensor<1xi64>,
    rhs_contracting_dimensions = dense<1> : tensor<1xi64>
  }} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = shapex.get_ranked_shape %0 : tensor<?x?x?xf32> -> !shapex.ranked_shape<[?,?,?]>
  return %1 : !shapex.ranked_shape<[?,?,?]>
}

// -----

// CHECK-LABEL: func @dynamic_reshape
func @dynamic_reshape(%arg0: tensor<?xf32>, %arg1: tensor<2xindex>) -> !shapex.ranked_shape<[?,?]> {
  // CHECK-DAG: %[[SHAPE:.+]] = "shapex.from_extent_tensor"(%arg1)
  // CHECK-DAG: return %[[SHAPE]]
  %0 = "mhlo.dynamic_reshape"(%arg0, %arg1) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %1 = shapex.get_ranked_shape %0 : tensor<?x?xf32> -> !shapex.ranked_shape<[?,?]>
  return %1 : !shapex.ranked_shape<[?,?]>
}

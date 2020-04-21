// RUN: iree-opt -split-input-file -verify-diagnostics -iree-shape-materialize-calculations -iree-shape-cleanup-placeholders %s | IreeFileCheck %s

// CHECK-LABEL: @compileTime
// CHECK-SAME: %[[T:[^:[:space:]]+]]: tensor<?x2xf32>
// CHECK-SAME: %[[SHAPE:[^:[:space:]]+]]: !shapex.ranked_shape<[?,2]>
func @compileTime(%arg0 : tensor<?x2xf32>, %arg1 : !shapex.ranked_shape<[?,2]>) -> (tensor<?x2xf32> , !shapex.ranked_shape<[?,2]>) {
  // CHECK-NOT: shapex.tie_shape
  // CHECK-NOT: shapex.get_ranked_shape
  %0 = shapex.tie_shape %arg0, %arg1 : tensor<?x2xf32>, !shapex.ranked_shape<[?,2]>
  // CHECK: %[[ABS:.+]] = "xla_hlo.abs"(%[[T]])
  // The only thing special about abs is that we have a compile time shape
  // calculation for it.
  %1 = "xla_hlo.abs"(%0) : (tensor<?x2xf32>) -> (tensor<?x2xf32>)
  %2 = shapex.get_ranked_shape %1 : tensor<?x2xf32> -> !shapex.ranked_shape<[?,2]>
  // CHECK: return %[[ABS]], %[[SHAPE]]
  return %1, %2 : tensor<?x2xf32>, !shapex.ranked_shape<[?,2]>
}

// -----

// CHECK-LABEL: @f
func @f(%arg0 : !shapex.ranked_shape<[?]>, %arg1 : !shapex.ranked_shape<[?]>) -> (!shapex.ranked_shape<[?]>) {
  // CHECK-DAG: %[[LHSEXTENT:.+]] = shapex.ranked_dim %arg0[0]
  // CHECK-DAG: %[[RHSEXTENT:.+]] = shapex.ranked_dim %arg1[0]
  // CHECK-DAG: %[[GT:.+]] = cmpi "ugt", %[[LHSEXTENT]], %[[RHSEXTENT]] : index
  // CHECK-DAG: %[[MAX:.+]] = select %[[GT]], %[[LHSEXTENT]], %[[RHSEXTENT]] : index
  // CHECK-DAG: %[[RS:.+]] = shapex.make_ranked_shape %[[MAX]]
  // CHECK-DAG: return %[[RS]]
  %0 = "shapex.ranked_broadcast_shape"(%arg0, %arg1) {
    lhs_broadcast_dimensions = dense<[0]> : tensor<1xi64>,
    rhs_broadcast_dimensions = dense<[0]> : tensor<1xi64>
  } : (!shapex.ranked_shape<[?]>, !shapex.ranked_shape<[?]>) -> !shapex.ranked_shape<[?]>
  return %0 : !shapex.ranked_shape<[?]>
}

// -----
// CHECK-LABEL: @runTimeFallback
// CHECK-SAME: %[[T:[^:[:space:]]+]]: tensor<?x2xf32>
// CHECK-SAME: %[[SHAPE:[^:[:space:]]+]]: !shapex.ranked_shape<[?,2]>
func @runTimeFallback(%arg0 : tensor<?x2xf32>, %arg1 : !shapex.ranked_shape<[?,2]>) -> (tensor<?x2xf32> , !shapex.ranked_shape<[?,2]>) {
  // CHECK-NOT: shapex.tie_shape
  // CHECK-NOT: shapex.get_ranked_shape
  %0 = shapex.tie_shape %arg0, %arg1 : tensor<?x2xf32>, !shapex.ranked_shape<[?,2]>
  // CHECK: %[[RESULT:.+]] = "unknown_op"
  // @expected-remark @+1 {{unable to materialize shape calculation}}
  %1 = "unknown_op"(%0) : (tensor<?x2xf32>) -> (tensor<?x2xf32>)
  // CHECK: %[[DIM:.+]] = dim %[[RESULT]], 0
  // CHECK: %[[RESULT_SHAPE:.+]] = shapex.make_ranked_shape %[[DIM]]
  // CHECK: return %[[RESULT]], %[[RESULT_SHAPE]]
  %2 = shapex.get_ranked_shape %1 : tensor<?x2xf32> -> !shapex.ranked_shape<[?,2]>
  return %1, %2 : tensor<?x2xf32>, !shapex.ranked_shape<[?,2]>
}

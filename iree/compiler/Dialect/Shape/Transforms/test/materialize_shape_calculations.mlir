// RUN: iree-opt -split-input-file -verify-diagnostics -iree-shape-materialize-calculations -iree-shape-cleanup-placeholders %s | IreeFileCheck %s

// CHECK-LABEL: @compileTime
// CHECK-SAME: %[[T:[^:[:space:]]+]]: tensor<?x2xf32>
// CHECK-SAME: %[[SHAPE:[^:[:space:]]+]]: !shapex.ranked_shape<[?,2]>
func @compileTime(%arg0 : tensor<?x2xf32>, %arg1 : !shapex.ranked_shape<[?,2]>) -> (tensor<?x2xf32> , !shapex.ranked_shape<[?,2]>) {
  // CHECK-NOT: shapex.tie_shape
  // CHECK-NOT: shapex.get_ranked_shape
  %0 = shapex.tie_shape %arg0, %arg1 : tensor<?x2xf32>, !shapex.ranked_shape<[?,2]>
  // CHECK: %[[ABS:.+]] = "mhlo.abs"(%[[T]])
  // The only thing special about abs is that we have a compile time shape
  // calculation for it.
  %1 = "mhlo.abs"(%0) : (tensor<?x2xf32>) -> (tensor<?x2xf32>)
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
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[RESULT:.+]] = "unknown_op"
  // @expected-remark @+1 {{unable to materialize shape calculation}}
  %1 = "unknown_op"(%0) : (tensor<?x2xf32>) -> (tensor<?x2xf32>)
  // CHECK: %[[DIM:.+]] = dim %[[RESULT]], %[[C0]]
  // CHECK: %[[RESULT_SHAPE:.+]] = shapex.make_ranked_shape %[[DIM]]
  // CHECK: return %[[RESULT]], %[[RESULT_SHAPE]]
  %2 = shapex.get_ranked_shape %1 : tensor<?x2xf32> -> !shapex.ranked_shape<[?,2]>
  return %1, %2 : tensor<?x2xf32>, !shapex.ranked_shape<[?,2]>
}

// -----
// CHECK-LABEL: func @f
func @f(%arg0: index, %arg1: index) -> (index, index, index) {
  %0 = shapex.make_ranked_shape %arg0, %arg1 : (index, index) -> !shapex.ranked_shape<[?,?]>
  %1 = "shapex.gather_extents"(%0) {indices = dense<[1, 1, 0]> : tensor<3xi64>} : (!shapex.ranked_shape<[?,?]>) -> !shapex.ranked_shape<[?,?,?]>
  %2:3 = shapex.ranked_dims %1 : !shapex.ranked_shape<[?,?,?]> -> index, index, index
  // CHECK: return %arg1, %arg1, %arg0
  return %2#0, %2#1, %2#2 : index, index, index
}

// -----
// CHECK-LABEL: func @f
func @f(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index, index) {
  %0 = shapex.make_ranked_shape %arg0 : (index) -> !shapex.ranked_shape<[?]>
  %1 = shapex.make_ranked_shape %arg1 : (index) -> !shapex.ranked_shape<[?]>
  %2 = shapex.make_ranked_shape %arg2 : (index) -> !shapex.ranked_shape<[?]>
  %gathered = "shapex.gather_extents"(%0, %1, %2) {indices = dense<[2, 2, 1, 0]> : tensor<4xi64>} : (!shapex.ranked_shape<[?]>, !shapex.ranked_shape<[?]>, !shapex.ranked_shape<[?]>) -> !shapex.ranked_shape<[?,?,?,?]>
  %extents:4 = shapex.ranked_dims %gathered : !shapex.ranked_shape<[?,?,?,?]> -> index, index, index, index
  // CHECK: return %arg2, %arg2, %arg1, %arg0
  return %extents#0, %extents#1, %extents#2, %extents#3 : index, index, index, index
}

// -----
// CHECK-LABEL: func @f
func @f(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index, index, index, index) {
  %0 = shapex.make_ranked_shape %arg0 : (index) -> !shapex.ranked_shape<[?,2]>
  %1 = shapex.make_ranked_shape %arg1 : (index) -> !shapex.ranked_shape<[3,?]>
  %2 = shapex.make_ranked_shape %arg2 : (index) -> !shapex.ranked_shape<[7,?]>
  %gathered = "shapex.gather_extents"(%0, %1, %2) {indices = dense<[0, 1, 2, 3, 4, 5]> : tensor<6xi64>} : (!shapex.ranked_shape<[?,2]>, !shapex.ranked_shape<[3,?]>, !shapex.ranked_shape<[7,?]>) -> !shapex.ranked_shape<[?,2,3,?,7,?]>
  %extents:6 = shapex.ranked_dims %gathered : !shapex.ranked_shape<[?,2,3,?,7,?]> -> index, index, index, index, index, index
  // CHECK: return %arg0, %c2{{.*}}, %c3{{.*}}, %arg1, %c7{{.*}}, %arg2
  return %extents#0, %extents#1, %extents#2, %extents#3, %extents#4, %extents#5 : index, index, index, index, index, index
}


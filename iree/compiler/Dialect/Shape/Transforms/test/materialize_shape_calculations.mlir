// RUN: iree-opt -split-input-file -verify-diagnostics -iree-shape-materialize-calculations -iree-shape-cleanup-placeholders %s | IreeFileCheck %s

// CHECK-LABEL: @compileTime
// CHECK-SAME: %[[T:[^:[:space:]]+]]: tensor<?x2xf32>
// CHECK-SAME: %[[SHAPE:[^:[:space:]]+]]: !shapex.ranked_shape<[?,2],i32>
func @compileTime(%arg0 : tensor<?x2xf32>, %arg1 : !shapex.ranked_shape<[?,2],i32>) -> (tensor<?x2xf32> , !shapex.ranked_shape<[?,2],i32>) {
  // CHECK-NOT: shapex.tie_shape
  // CHECK-NOT: shapex.get_ranked_shape
  %0 = shapex.tie_shape %arg0, %arg1 : tensor<?x2xf32>, !shapex.ranked_shape<[?,2],i32>
  // CHECK: %[[ABS:.+]] = "xla_hlo.abs"(%[[T]])
  // The only thing special about abs is that we have a compile time shape
  // calculation for it.
  %1 = "xla_hlo.abs"(%0) : (tensor<?x2xf32>) -> (tensor<?x2xf32>)
  %2 = shapex.get_ranked_shape %1 : tensor<?x2xf32> -> !shapex.ranked_shape<[?,2],i32>
  // CHECK: return %[[ABS]], %[[SHAPE]]
  return %1, %2 : tensor<?x2xf32>, !shapex.ranked_shape<[?,2],i32>
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

// RUN: iree-opt -split-input-file -verify-diagnostics -iree-shape-materialize-calculations -iree-shape-cleanup-placeholders %s | IreeFileCheck %s

// CHECK-LABEL: @unaryAbs
// CHECK-SAME: %[[T:[^:[:space:]]+]]: tensor<?x2xf32>
// CHECK-SAME: %[[SHAPE:[^:[:space:]]+]]: !shapex.ranked_shape<[?,2],i32>
func @unaryAbs(%arg0 : tensor<?x2xf32>, %arg1 : !shapex.ranked_shape<[?,2],i32>) -> (tensor<?x2xf32> , !shapex.ranked_shape<[?,2],i32>) {
  // CHECK-NOT: shapex.tie_shape
  // CHECK-NOT: shapex.get_ranked_shape
  %0 = shapex.tie_shape %arg0, %arg1 : tensor<?x2xf32>, !shapex.ranked_shape<[?,2],i32>
  // CHECK: %[[ABS:.+]] = "xla_hlo.abs"(%[[T]])
  %1 = "xla_hlo.abs"(%0) : (tensor<?x2xf32>) -> (tensor<?x2xf32>)
  %2 = shapex.get_ranked_shape %1 : tensor<?x2xf32> -> !shapex.ranked_shape<[?,2],i32>
  // CHECK: return %[[ABS]], %[[SHAPE]]
  return %1, %2 : tensor<?x2xf32>, !shapex.ranked_shape<[?,2],i32>
}

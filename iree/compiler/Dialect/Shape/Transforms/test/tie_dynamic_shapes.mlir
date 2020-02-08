// RUN: iree-opt -split-input-file -verify-diagnostics -iree-shape-tie-dynamic %s | IreeFileCheck %s

// CHECK-LABEL: @tieIntermediates
// CHECK-SAME: %[[T1:[^:[:space:]]+]]: tensor
// CHECK-SAME: %[[T2:[^:[:space:]]+]]: tensor
func @tieIntermediates(%arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-DAG: %[[SUM:.+]] = addf %[[T1]], %[[T2]]
  // CHECK-DAG: %[[RS:.+]] = shapex.get_ranked_shape %[[SUM]]
  // CHECK-DAG: %[[TIED_SUM:.+]] = shapex.tie_shape %[[SUM]], %[[RS]]
  // CHECK-DAG: return %[[TIED_SUM]]
  %0 = addf %arg1, %arg2 : tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----
// CHECK-LABEL: @noTieStatic
// CHECK-SAME: %[[T1:[^:[:space:]]+]]: tensor
// CHECK-SAME: %[[T2:[^:[:space:]]+]]: tensor
func @noTieStatic(%arg1 : tensor<1xf32>, %arg2 : tensor<1xf32>) -> tensor<1xf32> {
  // CHECK-NOT: shapex.tie_shape
  // CHECK: %[[SUM:.+]] = addf %[[T1]], %[[T2]]
  // CHECK: return %[[SUM]]
  %0 = addf %arg1, %arg2 : tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----
// CHECK-LABEL: @noTieUnranked
// CHECK-SAME: %[[T1:[^:[:space:]]+]]: tensor
// CHECK-SAME: %[[T2:[^:[:space:]]+]]: tensor
func @noTieUnranked(%arg1 : tensor<*xf32>, %arg2 : tensor<*xf32>) -> tensor<*xf32> {
  // CHECK-NOT: shapex.tie_shape
  // CHECK: %[[SUM:.+]] = addf %[[T1]], %[[T2]]
  // CHECK: return %[[SUM]]
  %0 = addf %arg1, %arg2 : tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: @noTieRedundant
// CHECK-SAME: %[[T1:[^:[:space:]]+]]: tensor
// CHECK-SAME: %[[T2:[^:[:space:]]+]]: tensor
// CHECK-SAME: %[[RS1:[^:[:space:]]+]]: !shapex.ranked_shape
func @noTieRedundant(%arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>, %arg3 : !shapex.ranked_shape<[?]>) -> tensor<?xf32> {
  // CHECK-DAG: %[[SUM:.+]] = addf %[[T1]], %[[T2]]
  // CHECK-DAG: %[[TIED_SUM:.+]] = shapex.tie_shape %[[SUM]], %[[RS1]]
  // CHECK-DAG: return %[[TIED_SUM]]
  %0 = addf %arg1, %arg2 : tensor<?xf32>
  %1 = shapex.tie_shape %0, %arg3 : tensor<?xf32>, !shapex.ranked_shape<[?]>
  return %1 : tensor<?xf32>
}

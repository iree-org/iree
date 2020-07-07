// RUN: iree-opt -split-input-file -iree-flow-dispatchability-analysis -iree-flow-identify-dispatch-regions2 %s | IreeFileCheck %s

// -----
// CHECK-LABEL: @singleDispatchWithShapes
// CHECK-SAME: %[[A0:[^:[:space:]]+]]: tensor<?x4xf32>,
// CHECK-SAME: %[[A1:[^:[:space:]]+]]: !shapex.ranked_shape<[?,4]>,
// CHECK-SAME: %[[A2:[^:[:space:]]+]]: !shapex.ranked_shape<[?,4]>
func @singleDispatchWithShapes(%arg0 : tensor<?x4xf32>,
    %arg1 : !shapex.ranked_shape<[?,4]>, %arg2 : !shapex.ranked_shape<[?,4]>) -> tensor<?x4xf32> {
  // Lead-in tie_shape should be preserved outside of the dispatch region.
  // CHECK: %[[TS0:.+]] = shapex.tie_shape %[[A0]], %[[A1]]
  %0 = shapex.tie_shape %arg0, %arg1 : tensor<?x4xf32>, !shapex.ranked_shape<[?,4]>
  // Fragility: The order of CA? derives from the algorithm and is
  // otherwise not load bearing. Since on a single line, this is difficult to
  // make generic.
  // CHECK: %[[R0:.+]] = flow.dispatch.region[%[[UNUSED_WORKLOAD:.+]] : index](
  // CHECK-SAME: %[[CA2:.+]] = %[[A2]] : !shapex.ranked_shape<[?,4]>,
  // CHECK-SAME: %[[CA0:.+]] = %[[TS0]] : tensor<?x4xf32>,
  // CHECK-SAME: %[[CA1:.+]] = %[[A1]] : !shapex.ranked_shape<[?,4]>)
    // Dispatch region should contain captured tie_shapes.
    // CHECK: %[[R1:.+]] = shapex.tie_shape %[[CA0]], %[[CA1]]
    // CHECK: %[[R2:.+]] = mhlo.add %[[R1]], %[[R1]]
    // CHECK: %[[R3:.+]] = shapex.tie_shape %[[R2]], %[[CA2]]
    // CHECK: flow.return %[[R3]]
  %1 = mhlo.add %0, %0 : tensor<?x4xf32>
  %2 = shapex.tie_shape %1, %arg2 : tensor<?x4xf32>, !shapex.ranked_shape<[?,4]>

  // Lead-out tie_shape should be preserved outside of the dispatch region.
  // CHECK: %[[R4:.+]] = shapex.tie_shape %[[R0]], %[[A2]]
  // CHECK: return %[[R4]]
  return %2 : tensor<?x4xf32>
}

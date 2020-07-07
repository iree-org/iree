// RUN: iree-opt -split-input-file -iree-flow-dispatchability-analysis -iree-flow-identify-dispatch-regions %s | IreeFileCheck %s

// CHECK-LABEL: @empty
func @empty() {
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: @simpleMath
func @simpleMath(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %[[WORKLOAD:.+]] = constant 4
  // CHECK-NEXT: %[[R1:.+]] = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD]] : index]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %1 = mhlo.add %arg1, %arg1 : tensor<4xf32>
  %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   flow.return %1 : tensor<4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[R1]] : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @stdElementwiseOps
func @stdElementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %[[WORKLOAD:.+]] = constant 4
  // CHECK-NEXT: %[[R1:.+]] = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD]] : index]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %1 = addf %arg1, %arg1 : tensor<4xf32>
  %0 = addf %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   %2 = subf %1, %arg1 : tensor<4xf32>
  %1 = subf %0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   %3 = mulf %2, %arg1 : tensor<4xf32>
  %2 = mulf %1, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[R1]] : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @hloElementwiseOps
func @hloElementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %[[WORKLOAD:.+]] = constant 4
  // CHECK-NEXT: %0 = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD]] : index]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %1 = mhlo.add %arg1, %arg1 : tensor<4xf32>
  %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   %2 = mhlo.subtract %1, %arg1 : tensor<4xf32>
  %1 = mhlo.subtract %0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   %3 = mhlo.multiply %2, %arg1 : tensor<4xf32>
  %2 = mhlo.multiply %1, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %0 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @interleavedDot
func @interleavedDot(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  // NOTE: Fragile ordering. Workload constants are emitted in order a the
  // top of the block.
  // CHECK: %[[WORKLOAD0:.+]] = constant 16 : index
  // CHECK: %[[WORKLOAD1:.+]] = constant 16 : index
  // CHECK: %[[WORKLOAD2:.+]] = constant 16 : index
  // CHECK: %[[R0:.+]] = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD0]] : index]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NEXT:   %3 = mhlo.add %arg1, %arg1 : tensor<4x4xf32>
  %0 = mhlo.add %arg0, %arg0 : tensor<4x4xf32>
  // CHECK-NEXT: flow.return %3 : tensor<4x4xf32>
  // CHECK-NEXT: }
  // CHECK: %[[R1:.+]] = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD1]] : index]
  // CHECK-SAME: (%arg1 = %[[R0]] : tensor<4x4xf32>, %arg2 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NEXT:   %3 = "mhlo.dot"(%arg1, %arg2) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = "mhlo.dot"(%0, %arg0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4x4xf32>
  // CHECK-NEXT: }
  // CHECK: %[[R2:.+]] = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD2]] : index]
  // CHECK-SAME: (%arg1 = %[[R1]] : tensor<4x4xf32>, %arg2 = %arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NEXT:   %3 = mhlo.multiply %arg1, %arg2 : tensor<4x4xf32>
  %2 = mhlo.multiply %1, %arg0 : tensor<4x4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4x4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[R2]] : tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @caller
func @caller(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %[[WORKLOAD0:.+]] = constant 4 : index
  // CHECK-NEXT: %[[R0:.+]] = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD0]] : index]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %1 = mhlo.add %arg1, %arg1 : tensor<4xf32>
  %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   %2 = call @callee(%1) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = call @callee(%0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   %3 = mhlo.multiply %2, %arg1 : tensor<4xf32>
  %2 = mhlo.multiply %1, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   flow.return %3 : tensor<4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[R0]] : tensor<4xf32>
  return %2 : tensor<4xf32>
}
// CHECK-LABEL: func @callee
func @callee(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[WORKLOAD0:.+]] = constant 4 : index
  // CHECK: %[[R0:.+]] = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD0]] : index]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %1 = mhlo.multiply %arg1, %arg1 : tensor<4xf32>
  %0 = mhlo.multiply %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT:   flow.return %1 : tensor<4xf32>
  // CHECK-NEXT: }
  // CHECK: return %[[R0]] : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @single_reduction
func @single_reduction(%arg0 : tensor<4x8xf32>) -> tensor<4xf32> {
  // CHECK-DAG: %[[INITIAL:.+]] = constant dense<0.000000e+00>
  %0 = constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[WORKLOAD0:.+]] = constant 4 : index
  // CHECK: %[[RESULT:.+]] = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD0]] : index]
  // CHECK-SAME: (%arg1 = %arg0 : tensor<4x8xf32>, %arg2 = %[[INITIAL]] : tensor<f32>) -> tensor<4xf32>
  // CHECK-NEXT: = "mhlo.reduce"(%arg1, %arg2)
  %1 = "mhlo.reduce"(%arg0, %0) ( {
  ^bb0(%arg1 : tensor<f32>, %arg2 : tensor<f32>):
    %2 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: flow.return
  // CHECK: return %[[RESULT]] : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @multi_reduction
func @multi_reduction(%arg0 : tensor<4x8xf32>, %arg1 : tensor<4x8xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  // CHECK-DAG: %[[INITIALA:.+]] = constant dense<0.000000e+00>
  %0 = constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[INITIALB:.+]] = constant dense<1.000000e+00>
  %1 = constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[WORKLOAD0:.+]] = constant 4 : index
  // CHECK: %[[RESULT:.+]]:2 = flow.dispatch.region
  // CHECK-SAME: [%[[WORKLOAD0]] : index]
  // CHECK-SAME: (%arg2 = %arg0 : tensor<4x8xf32>, %arg3 = %arg1 : tensor<4x8xf32>, %arg4 = %[[INITIALA]] : tensor<f32>, %arg5 = %[[INITIALB]] : tensor<f32>) -> (tensor<4xf32>, tensor<4xf32>)
  // CHECK-NEXT: = "mhlo.reduce"(%arg2, %arg3, %arg4, %arg5)
  %2, %3 = "mhlo.reduce"(%arg0, %arg1, %0, %1) ( {
  ^bb0(%arg0_lhs : tensor<f32>, %arg1_lhs : tensor<f32>, %arg0_rhs : tensor<f32>, %arg1_rhs : tensor<f32>):
    %4 = mhlo.add %arg0_lhs, %arg0_rhs : tensor<f32>
    %5 = mhlo.add %arg1_lhs, %arg1_rhs : tensor<f32>
    "mhlo.return"(%4, %5) : (tensor<f32>, tensor<f32>) -> ()
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<4xf32>, tensor<4xf32>)
  // CHECK: flow.return
  // CHECK: return %[[RESULT]]#0, %[[RESULT]]#1 : tensor<4xf32>, tensor<4xf32>
  return %2, %3 : tensor<4xf32>, tensor<4xf32>
}

// -----
// CHECK-LABEL: @singleDispatchWithShapes
// CHECK-SAME: %[[A0:[^:[:space:]]+]]: tensor<?x4xf32>,
// CHECK-SAME: %[[A1:[^:[:space:]]+]]: !shapex.ranked_shape<[?,4]>,
// CHECK-SAME: %[[A2:[^:[:space:]]+]]: !shapex.ranked_shape<[?,4]>
func @singleDispatchWithShapes(%arg0 : tensor<?x4xf32>,
    %rs0 : !shapex.ranked_shape<[?,4]>, %rs1 : !shapex.ranked_shape<[?,4]>) -> tensor<?x4xf32> {
  // Lead-in tie_shape should be preserved outside of the dispatch region.
  // CHECK: %[[TS0:.+]] = shapex.tie_shape %[[A0]], %[[A1]]
  %0 = shapex.tie_shape %arg0, %rs0 : tensor<?x4xf32>, !shapex.ranked_shape<[?,4]>
  // CHECK: %[[R0:.+]] = flow.dispatch.region[%[[UNUSED_WORKLOAD:.+]] : index](
  // CHECK-SAME: %[[CA0:.+]] = %[[TS0]] : tensor<?x4xf32>,
  // CHECK-SAME: %[[CA1:.+]] = %[[A1]] : !shapex.ranked_shape<[?,4]>,
  // CHECK-SAME: %[[CA2:.+]] = %[[A2]] : !shapex.ranked_shape<[?,4]>)
    // Dispatch region should contain captured tie_shapes.
    // CHECK: %[[R1:.+]] = shapex.tie_shape %[[CA0]], %[[CA1]]
    // CHECK: %[[R2:.+]] = mhlo.add %[[R1]], %[[R1]]
    // CHECK: %[[R3:.+]] = shapex.tie_shape %[[R2]], %[[CA2]]
    // CHECK: flow.return %[[R3]]
  %1 = mhlo.add %0, %0 : tensor<?x4xf32>
  %2 = shapex.tie_shape %1, %rs1 : tensor<?x4xf32>, !shapex.ranked_shape<[?,4]>

  // Lead-out tie_shape should be preserved outside of the dispatch region.
  // CHECK: %[[R4:.+]] = shapex.tie_shape %[[R0]], %[[A2]]
  // CHECK: return %[[R4]]
  return %2 : tensor<?x4xf32>
}

// -----
// CHECK-LABEL: @fusedDispatchWithShapes
// CHECK-SAME: %[[A0:[^:[:space:]]+]]: tensor<?x4xf32>
// CHECK-SAME: %[[A1:[^:[:space:]]+]]: !shapex.ranked_shape<[?,4]>
func @fusedDispatchWithShapes(%arg0 : tensor<?x4xf32>,
    %rs0 : !shapex.ranked_shape<[?,4]>) -> tensor<?x4xf32> {
  // Lead-in tie_shape should be preserved outside of the dispatch region.
  // CHECK: %[[TS0:.+]] = shapex.tie_shape %[[A0]], %[[A1]]
  // CHECK: %[[R0:.+]] = flow.dispatch.region[%[[UNUSED_WORKLOAD:.+]] : index](
  // CHECK-SAME: %[[CA0:.+]] = %[[TS0]] : tensor<?x4xf32>,
  // CHECK-SAME: %[[CA1:.+]] = %[[A1]] : !shapex.ranked_shape<[?,4]>
    // Dispatch region should contain captured tie_shapes.
    // CHECK: %[[R1:.+]] = shapex.tie_shape %[[CA0]], %[[CA1]]
    // CHECK: %[[R2:.+]] = mhlo.add %[[R1]], %[[R1]]
    // CHECK: %[[R3:.+]] = shapex.tie_shape %[[R2]], %[[CA1]]
    // CHECK: %[[R4:.+]] = mhlo.multiply %[[R3]], %[[R1]]
    // CHECK: %[[R5:.+]] = shapex.tie_shape %[[R4]], %[[CA1]]
    // CHECK: flow.return %[[R5]]

  %0 = shapex.tie_shape %arg0, %rs0 : tensor<?x4xf32>, !shapex.ranked_shape<[?,4]>
  %1 = mhlo.add %0, %0 : tensor<?x4xf32>
  %2 = shapex.tie_shape %1, %rs0 : tensor<?x4xf32>, !shapex.ranked_shape<[?,4]>
  %3 = mhlo.multiply %2, %0 : tensor<?x4xf32>
  %4 = shapex.tie_shape %3, %rs0 : tensor<?x4xf32>, !shapex.ranked_shape<[?,4]>

  // Lead-out tie_shape should be preserved outside of the dispatch region.
  // CHECK: %[[R6:.+]] = shapex.tie_shape %[[R0]], %[[A1]]
  // CHECK: return %[[R6]]
  return %4 : tensor<?x4xf32>
}

// -----
// CHECK-LABEL: @fusedDispatchRootWithShapes
// CHECK-SAME: %[[A0:[^:[:space:]]+]]:
// CHECK-SAME: %[[A1:[^:[:space:]]+]]:
// CHECK-SAME: %[[RS0:[^:[:space:]]+]]:
// CHECK-SAME: %[[RS1:[^:[:space:]]+]]:
// CHECK-SAME: %[[RS2:[^:[:space:]]+]]:
func @fusedDispatchRootWithShapes(%arg0 : tensor<?x4xf32>,
    %arg1 : tensor<4x?xf32>,
    %rs0 : !shapex.ranked_shape<[?,4]>,
    %rs1 : !shapex.ranked_shape<[4,?]>,
    %rs2 : !shapex.ranked_shape<[?,?]>) -> tensor<?x?xf32> {
  // Lead-in tie_shape should be preserved outside of the dispatch region.
  // CHECK: %[[TS0:.+]] = shapex.tie_shape %[[A0]], %[[RS0]]
  // CHECK: %[[TS1:.+]] = shapex.tie_shape %[[A1]], %[[RS1]]
  // CHECK: %[[R0:.+]] = flow.dispatch.region[%[[UNUSED_WORKLOAD:.+]] : index](
  // CHECK-SAME: {
  // Verify that the ties are preserved (relying on outlining tested previously)
    // CHECK-DAG: %[[DTS0:.+]] = shapex.tie_shape {{.+}}: tensor<?x4xf32>, !shapex.ranked_shape<[?,4]>
    // CHECK-DAG: %[[DTS1:.+]] = shapex.tie_shape {{.+}}: tensor<4x?xf32>, !shapex.ranked_shape<[4,?]>
    // CHECK-DAG: %[[DR0:.+]] = "mhlo.dot"(%[[DTS0]], %[[DTS1]])
    // CHECK-DAG: %[[DTS1:.+]] = shapex.tie_shape %[[DR0]], {{.+}}: tensor<?x?xf32>, !shapex.ranked_shape<[?,?]>
    // CHECK: flow.return %[[DTS1]]
  // CHECK: }
  // Lead-out tie_shape should be preserved.
  // CHECK: %[[R1:.+]] = shapex.tie_shape %[[R0]], %[[RS2]]
  // CHECK: return %[[R1]]
  %0 = shapex.tie_shape %arg0, %rs0 : tensor<?x4xf32>, !shapex.ranked_shape<[?,4]>
  %1 = shapex.tie_shape %arg1, %rs1 : tensor<4x?xf32>, !shapex.ranked_shape<[4,?]>
  %2 = "mhlo.dot"(%0, %1) : (tensor<?x4xf32>, tensor<4x?xf32>) -> tensor<?x?xf32>
  %3 = shapex.tie_shape %2, %rs2 : tensor<?x?xf32>, !shapex.ranked_shape<[?,?]>
  return %3 : tensor<?x?xf32>
}

// TODO(benvanik): windowed reduction.

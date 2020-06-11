// RUN: iree-opt -split-input-file -iree-flow-outline-dispatch-regions -canonicalize %s | IreeFileCheck %s
// NOTE: Most of the common cases for outlining are tested via
// transformation.mlir; however, this test performs some specific tests
// of corner cases that are easier to access at this level.

// CHECK-LABEL: @dynamicRankedShapeModule
// Verify that the outlined function properly expands shape dims
// Note that all but the entry shape ties/ops are removed.
// CHECK: flow.executable @dynamicRankedShape_ex_dispatch_0
// CHECK: func @dynamicRankedShape_ex_dispatch_0(%[[EXARG0:.+]]: tensor<7x?x24x?xf32>, %[[EXARG1:.+]]: index, %[[EXARG2:.+]]: index) -> tensor<?x?x1024xf32> {
// CHECK-DAG: %[[EXSHAPE0:.+]] = shapex.make_ranked_shape %[[EXARG1]], %[[EXARG2]]
// CHECK-DAG: %[[EXT0:.+]] = shapex.tie_shape %[[EXARG0]], %[[EXSHAPE0]]
// CHECK-DAG: %[[EXT1:.+]] = "some_kind_of_sum"(%[[EXT0]])
// CHECK-DAG: return %[[EXT1]]
// Verify that the generated flow.dispatch op properly inputs individual shape dims
// CHECK: func @dynamicRankedShape(%[[ARG0:.+]]: tensor<7x?x24x?xf32>)
// CHECK-DAG: %[[C1:.*]] = constant 1 : index
// CHECK-DAG: %[[C3:.*]] = constant 3 : index
// CHECK-DAG: %[[D1:.+]] = dim %[[ARG0]], %[[C1]]
// CHECK-DAG: %[[D3:.+]] = dim %[[ARG0]], %[[C3]]
// CHECK-DAG: %[[WORKLOAD0:.+]] = constant 1024 : index
// CHECK-DAG: %[[DISPATCH:.+]] = flow.dispatch @dynamicRankedShape_ex_dispatch_0::@dynamicRankedShape_ex_dispatch_0[%[[WORKLOAD0]] : index](%[[ARG0]], %[[D1]], %[[D3]]) : (tensor<7x?x24x?xf32>, index, index)
// CHECK-DAG: return %[[DISPATCH]]
module @dynamicRankedShapeModule {
func @dynamicRankedShape(%arg0 : tensor<7x?x24x?xf32>) -> tensor<?x?x1024xf32> {
  %c1 = constant 1 : index
  %c3 = constant 3 : index
  %dim1 = dim %arg0, %c1 : tensor<7x?x24x?xf32>
  %dim3 = dim %arg0, %c3 : tensor<7x?x24x?xf32>
  %workload0 = constant 1024 : index
  %shape0 = shapex.make_ranked_shape %dim1, %dim3 : (index, index) -> !shapex.ranked_shape<[7,?,24,?]>
  %1 = flow.dispatch.region[%workload0 : index](%arg1 = %arg0 : tensor<7x?x24x?xf32>, %arg2 = %shape0 : !shapex.ranked_shape<[7,?,24,?]>) -> tensor<?x?x1024xf32> {
    %2 = shapex.tie_shape %arg1, %arg2 : tensor<7x?x24x?xf32>, !shapex.ranked_shape<[7,?,24,?]>
    // Simulate a custom op that shuffles the input in a weird way.
    %3 = "some_kind_of_sum"(%2) : (tensor<7x?x24x?xf32>) -> tensor<?x?x1024xf32>
    %4 = shapex.ranked_dim %arg2[1] : !shapex.ranked_shape<[7,?,24,?]> -> index
    %5 = shapex.ranked_dim %arg2[3] : !shapex.ranked_shape<[7,?,24,?]> -> index
    %6 = shapex.make_ranked_shape %4, %5 : (index, index) -> !shapex.ranked_shape<[?,?,1024]>
    %7 = shapex.tie_shape %3, %6 : tensor<?x?x1024xf32>, !shapex.ranked_shape<[?,?,1024]>
    flow.return %7 : tensor<?x?x1024xf32>
  }
  return %1 : tensor<?x?x1024xf32>
}
}

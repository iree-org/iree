// RUN: iree-opt -split-input-file -iree-flow-outline-dispatch-regions -canonicalize %s | IreeFileCheck %s
// NOTE: Most of the common cases for outlining are tested via
// transformation.mlir; however, this test performs some specific tests
// of corner cases that are easier to access at this level.

// CHECK-LABEL: @dynamicRankedShapeModule
// Verify that the outlined function properly expands shape dims
// CHECK: flow.executable @dynamicRankedShape_ex_dispatch_0 {
// CHECK: func @dynamicRankedShape_ex_dispatch_0(%[[EXARG0:.+]]: tensor<7x?x24x?xf32>, %[[EXARG1:.+]]: index, %[[EXARG2:.+]]: index) -> tensor<1024xf32> {
// CHECK-DAG: %[[EXSHAPE0:.+]] = shapex.make_ranked_shape %[[EXARG1]], %[[EXARG2]]
// CHECK-DAG: %[[EXT0:.+]] = shapex.tie_shape %[[EXARG0]], %[[EXSHAPE0]]
// CHECK-DAG: %[[EXT1:.+]] = "some_kind_of_sum"(%[[EXT0]])
// Verify that the generated flow.dispatch op properly inputs individual shape dims
// CHECK: func @dynamicRankedShape(%[[ARG0:.+]]: tensor<7x?x24x?xf32>)
// CHECK-DAG: %[[D1:.+]] = dim %[[ARG0]], 1
// CHECK-DAG: %[[D3:.+]] = dim %[[ARG0]], 3
// CHECK-DAG: %[[WORKLOAD0:.+]] = constant 1024 : index
// CHECK-DAG: %[[DISPATCH:.+]] = flow.dispatch @dynamicRankedShape_ex_dispatch_0::@dynamicRankedShape_ex_dispatch_0[%[[WORKLOAD0]] : index](%[[ARG0]], %[[D1]], %[[D3]]) : (tensor<7x?x24x?xf32>, index, index)
// CHECK-DAG: return %[[DISPATCH]]
module @dynamicRankedShapeModule {
func @dynamicRankedShape(%arg0 : tensor<7x?x24x?xf32>) -> tensor<1024xf32> {
  %dim1 = dim %arg0, 1 : tensor<7x?x24x?xf32>
  %dim3 = dim %arg0, 3 : tensor<7x?x24x?xf32>
  %workload0 = constant 1024 : index
  %shape0 = shapex.make_ranked_shape %dim1, %dim3 -> !shapex.ranked_shape<[7,?,24,?]>
  %1 = flow.dispatch.region[%workload0 : index](%arg1 = %arg0 : tensor<7x?x24x?xf32>, %arg2 = %shape0 : !shapex.ranked_shape<[7,?,24,?]>) -> tensor<1024xf32> {
    %2 = shapex.tie_shape %arg1, %arg2 : tensor<7x?x24x?xf32>, !shapex.ranked_shape<[7,?,24,?]>
    %3 = "some_kind_of_sum"(%2) : (tensor<7x?x24x?xf32>) -> tensor<1024xf32>
    flow.return %3 : tensor<1024xf32>
  }
  return %1 : tensor<1024xf32>
}
}

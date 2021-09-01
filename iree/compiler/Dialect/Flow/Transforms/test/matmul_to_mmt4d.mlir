// RUN: iree-opt -split-input-file --iree-flow-convert-matmul-to-mmt4d %s | IreeFileCheck %s

func @check_mmt4d(%arg0: tensor<24x8xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<24x32xf32>) -> tensor<24x32xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x8xf32>, tensor<8x32xf32>) outs(%arg2 : tensor<24x32xf32>) -> tensor<24x32xf32>
    return %0 : tensor<24x32xf32>
}
// CHECK-DAG:#[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
// CHECK-DAG:#[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG:#[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d0, d3)>
//      CHECK: @check_mmt4d(%[[LHS:.+]]: tensor<24x8xf32>, %[[RHS:.+]]: tensor<8x32xf32>, %[[DST:.+]]: tensor<24x32xf32>)
//      CHECK: %[[LHS4D:.+]] = linalg.tensor_expand_shape %[[LHS]]
// CHECK-SAME:   tensor<24x8xf32> into tensor<6x4x2x4xf32>
//      CHECK: %[[RHS4D:.+]] = linalg.tensor_expand_shape %[[RHS]]
// CHECK-SAME:   tensor<8x32xf32> into tensor<2x4x8x4xf32>
//      CHECK: %[[DST4D:.+]] = linalg.tensor_expand_shape %[[DST]]
// CHECK-SAME:   tensor<24x32xf32> into tensor<6x4x8x4xf32>
//      CHECK: %[[LHS4DT_INIT:.+]] = linalg.init_tensor [6, 2, 4, 4] : tensor<6x2x4x4xf32>
//      CHECK: %[[LHS4DT:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:   ins(%[[LHS4D]] : tensor<6x4x2x4xf32>) outs(%[[LHS4DT_INIT]] : tensor<6x2x4x4xf32>) {
// CHECK-NEXT:     ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:       linalg.yield
// CHECK-NEXT:    } -> tensor<6x2x4x4xf32>
//      CHECK: %[[RHS4DT_INIT:.+]] = linalg.init_tensor [8, 2, 4, 4] : tensor<8x2x4x4xf32>
//      CHECK: %[[RHS4DT:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP2]], #[[MAP1]]],
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:   ins(%[[RHS4D]] : tensor<2x4x8x4xf32>) outs(%[[RHS4DT_INIT]] : tensor<8x2x4x4xf32>) {
// CHECK-NEXT:     ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:         linalg.yield %arg3 : f32
// CHECK-NEXT:   } -> tensor<8x2x4x4xf32>
// CHECK-NEXT: %[[DST4DT_INIT:.+]] = linalg.init_tensor [6, 8, 4, 4] : tensor<6x8x4x4xf32>
//      CHECK: %[[DST4DT:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
// CHECK-SAME:    ins(%[[DST4D]] : tensor<6x4x8x4xf32>) outs(%[[DST4DT_INIT]] : tensor<6x8x4x4xf32>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:          linalg.yield %arg3 : f32
// CHECK-NEXT:    } -> tensor<6x8x4x4xf32>
//      CHECK: %[[MMT4D:.+]] = linalg.mmt4d ins(%[[LHS4DT]], %[[RHS4DT]] : tensor<6x2x4x4xf32>, tensor<8x2x4x4xf32>) outs(%[[DST4DT]] : tensor<6x8x4x4xf32>) -> tensor<6x8x4x4xf32>
//      CHECK: %[[MMT4DT_INIT:.+]] = linalg.init_tensor [6, 4, 8, 4] : tensor<6x4x8x4xf32>
//      CHECK: %[[MMT4DT:.+]] = linalg.generic
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
// CHECK-SAME:    ins(%[[MMT4D]] : tensor<6x8x4x4xf32>) outs(%[[MMT4DT_INIT]] : tensor<6x4x8x4xf32>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:           linalg.yield %arg3 : f32
// CHECK-NEXT:    } -> tensor<6x4x8x4xf32>
//      CHECK: %[[RESULT:.+]] = linalg.tensor_collapse_shape %[[MMT4DT]]
// CHECK-SAME:    tensor<6x4x8x4xf32> into tensor<24x32xf32>
//      CHECK: return %[[RESULT]] : tensor<24x32xf32>

// -----
func @check_mmt4d_with_init_tensor_and_fill(%arg0: tensor<24x8xf32>, %arg1: tensor<8x32xf32>) -> tensor<24x32xf32> {
    %c0 = constant 0.0 : f32
    %0 = linalg.init_tensor [24, 32] : tensor<24x32xf32>
    %1 = linalg.fill(%c0, %0) : f32, tensor<24x32xf32> -> tensor<24x32xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<24x8xf32>, tensor<8x32xf32>) outs(%1 : tensor<24x32xf32>) -> tensor<24x32xf32>
    return %2 : tensor<24x32xf32>
}
// CHECK-DAG:#[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
// CHECK-DAG:#[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG:#[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d0, d3)>
//      CHECK: @check_mmt4d_with_init_tensor_and_fill(%[[LHS:.+]]: tensor<24x8xf32>, %[[RHS:.+]]: tensor<8x32xf32>)
//      CHECK: %[[ZERO:.+]] = constant 0.000000e+00 : f32
//      CHECK: %[[LHS4D:.+]] = linalg.tensor_expand_shape %[[LHS]]
// CHECK-SAME:   tensor<24x8xf32> into tensor<6x4x2x4xf32>
//      CHECK: %[[RHS4D:.+]] = linalg.tensor_expand_shape %[[RHS]]
// CHECK-SAME:   tensor<8x32xf32> into tensor<2x4x8x4xf32>
//      CHECK: %[[DST_INIT:.+]] = linalg.init_tensor [6, 8, 4, 4] : tensor<6x8x4x4xf32>
//      CHECK: [[DST:.+]] linalg.fill(%[[ZERO:.+]], %[[DST_INIT]])

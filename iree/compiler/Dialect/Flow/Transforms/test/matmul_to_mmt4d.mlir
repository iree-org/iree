// RUN: iree-opt -split-input-file --iree-flow-convert-linalg-matmul-to-mmt4d='M0=8 K0=2 N0=4' %s | IreeFileCheck %s

func @check_mmt4d_f32_static_nopad(%arg0: tensor<24x8xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<24x32xf32>) -> tensor<24x32xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x8xf32>, tensor<8x32xf32>) outs(%arg2 : tensor<24x32xf32>) -> tensor<24x32xf32>
    return %0 : tensor<24x32xf32>
}
// CHECK-DAG:#[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
// CHECK-DAG:#[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG:#[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>
//      CHECK: @check_mmt4d_f32_static_nopad(%[[LHS:.+]]: tensor<24x8xf32>, %[[RHS:.+]]: tensor<8x32xf32>, %[[DST:.+]]: tensor<24x32xf32>)
//      CHECK: %[[LHS4D:.+]] = tensor.expand_shape %[[LHS]]
// CHECK-SAME:   tensor<24x8xf32> into tensor<3x8x4x2xf32>
//      CHECK: %[[RHS4D:.+]] = tensor.expand_shape %[[RHS]]
// CHECK-SAME:   tensor<8x32xf32> into tensor<4x2x8x4xf32>
//      CHECK: %[[DST4D:.+]] = tensor.expand_shape %[[DST]]
// CHECK-SAME:   tensor<24x32xf32> into tensor<3x8x8x4xf32>
//      CHECK: %[[LHS4DT_INIT:.+]] = linalg.init_tensor [3, 4, 8, 2] : tensor<3x4x8x2xf32>
//      CHECK: %[[LHS4DT:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:   ins(%[[LHS4D]] : tensor<3x8x4x2xf32>) outs(%[[LHS4DT_INIT]] : tensor<3x4x8x2xf32>) {
// CHECK-NEXT:     ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:       linalg.yield
// CHECK-NEXT:    } -> tensor<3x4x8x2xf32>
//      CHECK: %[[RHS4DT_INIT:.+]] = linalg.init_tensor [8, 4, 4, 2] : tensor<8x4x4x2xf32>
//      CHECK: %[[RHS4DT:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP2]], #[[MAP1]]],
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:   ins(%[[RHS4D]] : tensor<4x2x8x4xf32>) outs(%[[RHS4DT_INIT]] : tensor<8x4x4x2xf32>) {
// CHECK-NEXT:     ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:         linalg.yield %arg3 : f32
// CHECK-NEXT:   } -> tensor<8x4x4x2xf32>
// CHECK-NEXT: %[[DST4DT_INIT:.+]] = linalg.init_tensor [3, 8, 8, 4] : tensor<3x8x8x4xf32>
//      CHECK: %[[DST4DT:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
// CHECK-SAME:    ins(%[[DST4D]] : tensor<3x8x8x4xf32>) outs(%[[DST4DT_INIT]] : tensor<3x8x8x4xf32>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:          linalg.yield %arg3 : f32
// CHECK-NEXT:    } -> tensor<3x8x8x4xf32>
//      CHECK: %[[MMT4D:.+]] = linalg.mmt4d ins(%[[LHS4DT]], %[[RHS4DT]] : tensor<3x4x8x2xf32>, tensor<8x4x4x2xf32>) outs(%[[DST4DT]] : tensor<3x8x8x4xf32>) -> tensor<3x8x8x4xf32>
//      CHECK: %[[MMT4DT_INIT:.+]] = linalg.init_tensor [3, 8, 8, 4] : tensor<3x8x8x4xf32>
//      CHECK: %[[MMT4DT:.+]] = linalg.generic
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
// CHECK-SAME:    ins(%[[MMT4D]] : tensor<3x8x8x4xf32>) outs(%[[MMT4DT_INIT]] : tensor<3x8x8x4xf32>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:           linalg.yield %arg3 : f32
// CHECK-NEXT:    } -> tensor<3x8x8x4xf32>
//      CHECK: %[[RESULT:.+]] = tensor.collapse_shape %[[MMT4DT]]
// CHECK-SAME:    tensor<3x8x8x4xf32> into tensor<24x32xf32>
//      CHECK: return %[[RESULT]] : tensor<24x32xf32>

// -----
func @check_mmt4d_with_init_tensor_and_fill(%arg0: tensor<24x8xf32>, %arg1: tensor<8x32xf32>) -> tensor<24x32xf32> {
    %c0 = arith.constant 0.0 : f32
    %0 = linalg.init_tensor [24, 32] : tensor<24x32xf32>
    %1 = linalg.fill(%c0, %0) : f32, tensor<24x32xf32> -> tensor<24x32xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<24x8xf32>, tensor<8x32xf32>) outs(%1 : tensor<24x32xf32>) -> tensor<24x32xf32>
    return %2 : tensor<24x32xf32>
}
// CHECK-DAG:#[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
// CHECK-DAG:#[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG:#[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>
//      CHECK: @check_mmt4d_with_init_tensor_and_fill(%[[LHS:.+]]: tensor<24x8xf32>, %[[RHS:.+]]: tensor<8x32xf32>)
//      CHECK: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
//      CHECK: %[[LHS4D:.+]] = tensor.expand_shape %[[LHS]]
// CHECK-SAME:   tensor<24x8xf32> into tensor<3x8x4x2xf32>
//      CHECK: %[[RHS4D:.+]] = tensor.expand_shape %[[RHS]]
// CHECK-SAME:   tensor<8x32xf32> into tensor<4x2x8x4xf32>
//      CHECK: %[[DST_INIT:.+]] = linalg.init_tensor [3, 8, 8, 4] : tensor<3x8x8x4xf32>
//      CHECK: [[DST:.+]] linalg.fill(%[[ZERO:.+]], %[[DST_INIT]])

// -----
func @check_mmt4d_i8_static_pad(%arg0: tensor<3x5xi8>, %arg1: tensor<5x2xi8>, %arg2: tensor<3x2xi32>) -> tensor<3x2xi32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<3x5xi8>, tensor<5x2xi8>) outs(%arg2 : tensor<3x2xi32>) -> tensor<3x2xi32>
    return %0 : tensor<3x2xi32>
}
//CHECK-LABEL: @check_mmt4d_i8_static_pad(
// CHECK-SAME: %[[LHS:.+]]: tensor<3x5xi8>, %[[RHS:.+]]: tensor<5x2xi8>, %[[ACC:.+]]: tensor<3x2xi32>)
//      CHECK: %[[LHSPAD:.+]] = linalg.pad_tensor %[[LHS]] low[0, 0] high[5, 1]
//      CHECK: tensor<3x5xi8> to tensor<8x6xi8>
//      CHECK: %[[RHSPAD:.+]] = linalg.pad_tensor %[[RHS]] low[0, 0] high[1, 2]
//      CHECK: tensor<5x2xi8> to tensor<6x4xi8>
//      CHECK: %[[ACCPAD:.+]] = linalg.pad_tensor %[[ACC]] low[0, 0] high[5, 2]
//      CHECK: tensor<3x2xi32> to tensor<8x4xi32>
//      CHECK: %[[LHS4D:.+]] = tensor.expand_shape %[[LHSPAD]]
// CHECK-SAME: tensor<8x6xi8> into tensor<1x8x3x2xi8>
//      CHECK: %[[RHS4D:.+]] = tensor.expand_shape %[[RHSPAD]]
// CHECK-SAME: tensor<6x4xi8> into tensor<3x2x1x4xi8>
//      CHECK: %[[ACC4D:.+]] = tensor.expand_shape %[[ACCPAD]]
// CHECK-SAME: tensor<8x4xi32> into tensor<1x8x1x4xi32>
//  ... After the above padding, we are reduced to the same stuff as we have
//  ... already checked in the above testcases, so we skip checking that again.
//      CHECK: %[[RESPAD:.+]] = tensor.collapse_shape
// CHECK-SAME: tensor<1x8x1x4xi32> into tensor<8x4xi32>
//      CHECK: %[[RES:.+]] = tensor.extract_slice %[[RESPAD]][0, 0] [3, 2] [1, 1]
// CHECK-SAME: tensor<8x4xi32> to tensor<3x2xi32>
//      CEHCK: return %[[RES]] : tensor<3x2xi32>

// -----
func @check_mmt4d_i8_dynamic(%arg0: tensor<?x?xi8>, %arg1: tensor<?x?xi8>, %arg2: tensor<?x?xi32>) -> tensor<?x?xi32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xi8>, tensor<?x?xi8>) outs(%arg2 : tensor<?x?xi32>) -> tensor<?x?xi32>
    return %0 : tensor<?x?xi32>
}
//CHECK-LABEL: @check_mmt4d_i8_dynamic(
// CHECK-SAME: %[[LHS:.+]]: tensor<?x?xi8>, %[[RHS:.+]]: tensor<?x?xi8>, %[[ACC:.+]]: tensor<?x?xi32>)
// ... We omit checking the arithmetic computing padding amounts because that would
// ... be testing too fine details and that is tested already by end-to-end matmul tests.
//      CHECK: %[[LHSPAD:.+]] = linalg.pad_tensor %[[LHS]] low[0, 0] high[
//      CHECK: tensor<?x?xi8> to tensor<?x?xi8>
//      CHECK: %[[RHSPAD:.+]] = linalg.pad_tensor %[[RHS]] low[0, 0] high[
//      CHECK: tensor<?x?xi8> to tensor<?x?xi8>
//      CHECK: %[[ACCPAD:.+]] = linalg.pad_tensor %[[ACC]] low[0, 0] high[
//      CHECK: tensor<?x?xi32> to tensor<?x?xi32>
//      CHECK: %[[LHS4D:.+]] = tensor.expand_shape %[[LHSPAD]]
// CHECK-SAME: tensor<?x?xi8> into tensor<?x8x?x2xi8>
//      CHECK: %[[RHS4D:.+]] = tensor.expand_shape %[[RHSPAD]]
// CHECK-SAME: tensor<?x?xi8> into tensor<?x2x?x4xi8>
//      CHECK: %[[ACC4D:.+]] = tensor.expand_shape %[[ACCPAD]]
// CHECK-SAME: tensor<?x?xi32> into tensor<?x8x?x4xi32>
//  ... After the above padding, we are reduced to the same stuff as we have
//  ... already checked in the above testcases, so we skip checking that again.
//      CHECK: %[[RESPAD:.+]] = tensor.collapse_shape
// CHECK-SAME: tensor<?x8x?x4xi32> into tensor<?x?xi32>
//      CHECK: %[[RES:.+]] = tensor.extract_slice %[[RESPAD]][0, 0] [{{.*}}] [1, 1]
// CHECK-SAME: tensor<?x?xi32> to tensor<?x?xi32>
//      CEHCK: return %[[RES]] : tensor<?x?xi32>

// RUN: iree-opt -split-input-file -iree-flow-pad-linalg-ops %s | IreeFileCheck %s

func @matmul_f32_11x13x17(%lhs: tensor<11x17xf32>, %rhs: tensor<17x13xf32>, %init: tensor<11x13xf32>) -> tensor<11x13xf32> {
    %result = linalg.matmul ins(%lhs, %rhs : tensor<11x17xf32>, tensor<17x13xf32>) outs(%init : tensor<11x13xf32>) -> tensor<11x13xf32>
    return %result : tensor<11x13xf32>
}
// CHECK-LABEL: @matmul_f32_11x13x17
//  CHECK-SAME:   %[[LHS:.+]]: tensor<11x17xf32>
//  CHECK-SAME:   %[[RHS:.+]]: tensor<17x13xf32>
//  CHECK-SAME:   %[[DST:.+]]: tensor<11x13xf32>
//   CHECK-DAG:      %[[PADDED_LHS:.+]] = linalg.pad_tensor %[[LHS]]
//   CHECK-DAG:      %[[PADDED_RHS:.+]] = linalg.pad_tensor %[[RHS]]
//   CHECK-DAG:      %[[PADDED_DST:.+]] = linalg.pad_tensor %[[DST]]
//       CHECK:      %[[PADDED_RESULT:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[PADDED_LHS]], %[[PADDED_RHS]] : tensor<12x20xf32>, tensor<20x16xf32>)
//  CHECK-SAME:         outs(%[[PADDED_DST]] : tensor<12x16xf32>)
//       CHECK:      %[[RESULT:.+]] = tensor.extract_slice %[[PADDED_RESULT]][0, 0] [11, 13] [1, 1] : tensor<12x16xf32> to tensor<11x13xf32>
//       CHECK:      return %[[RESULT]] : tensor<11x13xf32>

// -----

func @matmul_f32_12x12x17(%lhs: tensor<12x17xf32>, %rhs: tensor<17x12xf32>, %init: tensor<12x12xf32>) -> tensor<12x12xf32> {
    %result = linalg.matmul ins(%lhs, %rhs : tensor<12x17xf32>, tensor<17x12xf32>) outs(%init : tensor<12x12xf32>) -> tensor<12x12xf32>
    return %result : tensor<12x12xf32>
}
// CHECK-LABEL: @matmul_f32_12x12x17
//  CHECK-SAME:   %[[LHS:.+]]: tensor<12x17xf32>
//  CHECK-SAME:   %[[RHS:.+]]: tensor<17x12xf32>
//  CHECK-SAME:   %[[DST:.+]]: tensor<12x12xf32>
//   CHECK-DAG:      %[[PADDED_LHS:.+]] = linalg.pad_tensor %[[LHS]]
//   CHECK-DAG:      %[[PADDED_RHS:.+]] = linalg.pad_tensor %[[RHS]]
//       CHECK:      %[[RESULT:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[PADDED_LHS]], %[[PADDED_RHS]] : tensor<12x20xf32>, tensor<20x12xf32>)
//  CHECK-SAME:         outs(%[[DST]] : tensor<12x12xf32>)
//       CHECK:      return %[[RESULT]] : tensor<12x12xf32>


// -----

func @matmul_i8_i8_i32_2x2x4(%lhs: tensor<2x4xi8>, %rhs: tensor<4x2xi8>, %dst: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %result = linalg.matmul ins(%lhs, %rhs : tensor<2x4xi8>, tensor<4x2xi8>) outs(%dst: tensor<2x2xi32>) -> tensor<2x2xi32>
    return %result : tensor<2x2xi32>
}
// CHECK-LABEL: @matmul_i8_i8_i32_2x2x4
//  CHECK-SAME:   %[[LHS:.+]]: tensor<2x4xi8>
//  CHECK-SAME:   %[[RHS:.+]]: tensor<4x2xi8>
//  CHECK-SAME:   %[[DST:.+]]: tensor<2x2xi32>
//   CHECK-DAG:      %[[PADDED_LHS:.+]] = linalg.pad_tensor %[[LHS]]
//   CHECK-DAG:      %[[PADDED_RHS:.+]] = linalg.pad_tensor %[[RHS]]
//   CHECK-DAG:      %[[PADDED_DST:.+]] = linalg.pad_tensor %[[DST]]
//       CHECK:      %[[PADDED_RESULT:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[PADDED_LHS]], %[[PADDED_RHS]] : tensor<4x4xi8>, tensor<4x4xi8>)
//  CHECK-SAME:         outs(%[[PADDED_DST]] : tensor<4x4xi32>)
//       CHECK:      %[[RESULT:.+]] = tensor.extract_slice %[[PADDED_RESULT]]
//       CHECK:      return %[[RESULT]] : tensor<2x2xi32>

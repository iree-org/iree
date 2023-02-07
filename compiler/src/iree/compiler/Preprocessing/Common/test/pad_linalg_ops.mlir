// RUN: iree-opt --split-input-file --iree-preprocessing-pad-linalg-ops %s | FileCheck %s

func.func @matmul_f32_11x13x17(%lhs: tensor<11x17xf32>, %rhs: tensor<17x13xf32>, %init: tensor<11x13xf32>) -> tensor<11x13xf32> {
    %result = linalg.matmul ins(%lhs, %rhs : tensor<11x17xf32>, tensor<17x13xf32>) outs(%init : tensor<11x13xf32>) -> tensor<11x13xf32>
    return %result : tensor<11x13xf32>
}
// CHECK-LABEL: @matmul_f32_11x13x17
//  CHECK-SAME:   %[[LHS:.+]]: tensor<11x17xf32>
//  CHECK-SAME:   %[[RHS:.+]]: tensor<17x13xf32>
//  CHECK-SAME:   %[[DST:.+]]: tensor<11x13xf32>
//   CHECK-DAG:      %[[PADDED_LHS:.+]] = tensor.pad %[[LHS]]
//   CHECK-DAG:      %[[PADDED_RHS:.+]] = tensor.pad %[[RHS]]
//   CHECK-DAG:      %[[PADDED_DST:.+]] = tensor.pad %[[DST]]
//       CHECK:      %[[PADDED_RESULT:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[PADDED_LHS]], %[[PADDED_RHS]] : tensor<12x20xf32>, tensor<20x16xf32>)
//  CHECK-SAME:         outs(%[[PADDED_DST]] : tensor<12x16xf32>)
//       CHECK:      %[[RESULT:.+]] = tensor.extract_slice %[[PADDED_RESULT]][0, 0] [11, 13] [1, 1] : tensor<12x16xf32> to tensor<11x13xf32>
//       CHECK:      return %[[RESULT]] : tensor<11x13xf32>

// -----

func.func @matmul_f32_12x12x17(%lhs: tensor<12x17xf32>, %rhs: tensor<17x12xf32>, %init: tensor<12x12xf32>) -> tensor<12x12xf32> {
    %result = linalg.matmul ins(%lhs, %rhs : tensor<12x17xf32>, tensor<17x12xf32>) outs(%init : tensor<12x12xf32>) -> tensor<12x12xf32>
    return %result : tensor<12x12xf32>
}
// CHECK-LABEL: @matmul_f32_12x12x17
//  CHECK-SAME:   %[[LHS:.+]]: tensor<12x17xf32>
//  CHECK-SAME:   %[[RHS:.+]]: tensor<17x12xf32>
//  CHECK-SAME:   %[[DST:.+]]: tensor<12x12xf32>
//   CHECK-DAG:      %[[PADDED_LHS:.+]] = tensor.pad %[[LHS]]
//   CHECK-DAG:      %[[PADDED_RHS:.+]] = tensor.pad %[[RHS]]
//       CHECK:      %[[RESULT:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[PADDED_LHS]], %[[PADDED_RHS]] : tensor<12x20xf32>, tensor<20x12xf32>)
//  CHECK-SAME:         outs(%[[DST]] : tensor<12x12xf32>)
//       CHECK:      return %[[RESULT]] : tensor<12x12xf32>


// -----

func.func @matmul_i8_i8_i32_2x2x4(%lhs: tensor<2x4xi8>, %rhs: tensor<4x2xi8>, %dst: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %result = linalg.matmul ins(%lhs, %rhs : tensor<2x4xi8>, tensor<4x2xi8>) outs(%dst: tensor<2x2xi32>) -> tensor<2x2xi32>
    return %result : tensor<2x2xi32>
}
// CHECK-LABEL: @matmul_i8_i8_i32_2x2x4
//  CHECK-SAME:   %[[LHS:.+]]: tensor<2x4xi8>
//  CHECK-SAME:   %[[RHS:.+]]: tensor<4x2xi8>
//  CHECK-SAME:   %[[DST:.+]]: tensor<2x2xi32>
//   CHECK-DAG:      %[[PADDED_LHS:.+]] = tensor.pad %[[LHS]]
//   CHECK-DAG:      %[[PADDED_RHS:.+]] = tensor.pad %[[RHS]]
//   CHECK-DAG:      %[[PADDED_DST:.+]] = tensor.pad %[[DST]]
//       CHECK:      %[[PADDED_RESULT:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[PADDED_LHS]], %[[PADDED_RHS]] : tensor<4x4xi8>, tensor<4x4xi8>)
//  CHECK-SAME:         outs(%[[PADDED_DST]] : tensor<4x4xi32>)
//       CHECK:      %[[RESULT:.+]] = tensor.extract_slice %[[PADDED_RESULT]]
//       CHECK:      return %[[RESULT]] : tensor<2x2xi32>

// -----

func.func @batch_matmul_f32_7x11x13x17(%lhs: tensor<7x11x17xf32>, %rhs: tensor<7x17x13xf32>, %init: tensor<7x11x13xf32>) -> tensor<7x11x13xf32> {
    %result = linalg.batch_matmul ins(%lhs, %rhs : tensor<7x11x17xf32>, tensor<7x17x13xf32>) outs(%init : tensor<7x11x13xf32>) -> tensor<7x11x13xf32>
    return %result : tensor<7x11x13xf32>
}

// CHECK-LABEL: func.func @batch_matmul_f32_7x11x13x17
//  CHECK-SAME: (%[[LHS:.+]]: tensor<7x11x17xf32>, %[[RHS:.+]]: tensor<7x17x13xf32>, %[[DST:.+]]: tensor<7x11x13xf32>)
//       CHECK:   %[[PADDED_LHS:.+]] = tensor.pad %[[LHS]] low[0, 0, 0] high[0, 1, 3]
//       CHECK:   %[[PADDED_RHS:.+]] = tensor.pad %[[RHS]] low[0, 0, 0] high[0, 3, 3]
//       CHECK:   %[[PADDED_DST:.+]] = tensor.pad %[[DST]] low[0, 0, 0] high[0, 1, 3]
//       CHECK:   %[[BM:.+]] = linalg.batch_matmul
//  CHECK-SAME:     ins(%[[PADDED_LHS]], %[[PADDED_RHS]] : tensor<7x12x20xf32>, tensor<7x20x16xf32>)
//  CHECK-SAME:     outs(%[[PADDED_DST]] : tensor<7x12x16xf32>)
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[BM]][0, 0, 0] [7, 11, 13] [1, 1, 1]
//       CHECK:   return %[[EXTRACT]]

// -----

func.func @batch_matmul_f32_7x12x12x17(%lhs: tensor<7x12x17xf32>, %rhs: tensor<7x17x12xf32>, %init: tensor<7x12x12xf32>) -> tensor<7x12x12xf32> {
    %result = linalg.batch_matmul ins(%lhs, %rhs : tensor<7x12x17xf32>, tensor<7x17x12xf32>) outs(%init : tensor<7x12x12xf32>) -> tensor<7x12x12xf32>
    return %result : tensor<7x12x12xf32>
}

// CHECK-LABEL: func.func @batch_matmul_f32_7x12x12x17
//  CHECK-SAME: (%[[LHS:.+]]: tensor<7x12x17xf32>, %[[RHS:.+]]: tensor<7x17x12xf32>, %[[DST:.+]]: tensor<7x12x12xf32>)
//       CHECK:   %[[PADDED_LHS:.+]] = tensor.pad %[[LHS]] low[0, 0, 0] high[0, 0, 3]
//       CHECK:   %[[PADDED_RHS:.+]] = tensor.pad %[[RHS]] low[0, 0, 0] high[0, 3, 0]
//       CHECK:   %[[BM:.+]] = linalg.batch_matmul
//  CHECK-SAME:     ins(%[[PADDED_LHS]], %[[PADDED_RHS]] : tensor<7x12x20xf32>, tensor<7x20x12xf32>)
//  CHECK-SAME:     outs(%[[DST]] : tensor<7x12x12xf32>)
//       CHECK:   return %[[BM]]

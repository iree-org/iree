// RUN: iree-opt --iree-codegen-config-tracking-canonicalize --split-input-file %s | FileCheck %s

// Test folding tensor.insert_slice into tensor.empty when sizes match (static).

// CHECK-LABEL: func.func @fold_full_insert_slice_static
//  CHECK-SAME:   %[[SRC:.*]]: tensor<4x8xf32>
//       CHECK:   return %[[SRC]] : tensor<4x8xf32>
func.func @fold_full_insert_slice_static(%src: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %empty = tensor.empty() : tensor<4x8xf32>
  %inserted = tensor.insert_slice %src into %empty[0, 0][4, 8][1, 1] : tensor<4x8xf32> into tensor<4x8xf32>
  return %inserted : tensor<4x8xf32>
}

// -----

// Test folding tensor.insert_slice into tensor.empty when sizes match (dynamic).

// CHECK-LABEL: func.func @fold_full_insert_slice_dynamic
//  CHECK-SAME:   %[[SRC:.*]]: tensor<?x128xf32>
//  CHECK-SAME:   %[[D:.*]]: index
//       CHECK:   return %[[SRC]] : tensor<?x128xf32>
func.func @fold_full_insert_slice_dynamic(%src: tensor<?x128xf32>, %d: index) -> tensor<?x128xf32> {
  %empty = tensor.empty(%d) : tensor<?x128xf32>
  %inserted = tensor.insert_slice %src into %empty[0, 0][%d, 128][1, 1] : tensor<?x128xf32> into tensor<?x128xf32>
  return %inserted : tensor<?x128xf32>
}

// -----

// Test that we don't fold when offsets are non-zero.

// CHECK-LABEL: func.func @no_fold_nonzero_offset
//       CHECK:   tensor.empty
//       CHECK:   tensor.insert_slice
func.func @no_fold_nonzero_offset(%src: tensor<4x8xf32>) -> tensor<8x8xf32> {
  %empty = tensor.empty() : tensor<8x8xf32>
  %inserted = tensor.insert_slice %src into %empty[4, 0][4, 8][1, 1] : tensor<4x8xf32> into tensor<8x8xf32>
  return %inserted : tensor<8x8xf32>
}

// -----

// Test that we don't fold when sizes don't match.

// CHECK-LABEL: func.func @no_fold_size_mismatch
//       CHECK:   tensor.empty
//       CHECK:   tensor.insert_slice
func.func @no_fold_size_mismatch(%src: tensor<4x8xf32>) -> tensor<8x16xf32> {
  %empty = tensor.empty() : tensor<8x16xf32>
  %inserted = tensor.insert_slice %src into %empty[0, 0][4, 8][1, 1] : tensor<4x8xf32> into tensor<8x16xf32>
  return %inserted : tensor<8x16xf32>
}

// -----

// Test that we don't fold when destination is not tensor.empty.
// Note: The standard canonicalizer already folds this case since offsets are
// zero and sizes match, so we check that the fold happens.

// CHECK-LABEL: func.func @fold_not_empty_but_full_slice
//  CHECK-SAME:   %[[SRC:[a-zA-Z0-9_]+]]: tensor<4x8xf32>
//       CHECK:   return %[[SRC]] : tensor<4x8xf32>
func.func @fold_not_empty_but_full_slice(%src: tensor<4x8xf32>, %dest: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %inserted = tensor.insert_slice %src into %dest[0, 0][4, 8][1, 1] : tensor<4x8xf32> into tensor<4x8xf32>
  return %inserted : tensor<4x8xf32>
}

// -----

// Test that we don't fold when strides are not 1.

// CHECK-LABEL: func.func @no_fold_non_unit_strides
//       CHECK:   tensor.empty
//       CHECK:   tensor.insert_slice
func.func @no_fold_non_unit_strides(%src: tensor<2x4xf32>) -> tensor<4x8xf32> {
  %empty = tensor.empty() : tensor<4x8xf32>
  %inserted = tensor.insert_slice %src into %empty[0, 0][2, 4][2, 2] : tensor<2x4xf32> into tensor<4x8xf32>
  return %inserted : tensor<4x8xf32>
}

// -----

// Test folding with result type requiring cast (different tensor types).

// CHECK-LABEL: func.func @fold_with_cast
//  CHECK-SAME:   %[[SRC:.*]]: tensor<4x8xf32>
//       CHECK:   %[[CAST:.*]] = tensor.cast %[[SRC]] : tensor<4x8xf32> to tensor<?x?xf32>
//       CHECK:   return %[[CAST]] : tensor<?x?xf32>
func.func @fold_with_cast(%src: tensor<4x8xf32>) -> tensor<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %empty = tensor.empty(%c4, %c8) : tensor<?x?xf32>
  %inserted = tensor.insert_slice %src into %empty[0, 0][4, 8][1, 1] : tensor<4x8xf32> into tensor<?x?xf32>
  return %inserted : tensor<?x?xf32>
}

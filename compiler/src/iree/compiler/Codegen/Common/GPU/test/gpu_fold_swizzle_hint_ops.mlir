// RUN: iree-opt --mlir-print-local-scope --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-apply-tiling-level, canonicalize, cse))" %s | FileCheck %s

// Test: tensor.extract_slice of swizzle_hint(tensor.empty) should fold
// to swizzle_hint(tensor.empty) with the sliced shape.
func.func @fold_extract_slice_of_swizzle_hint() -> tensor<16x32xf32> {
  %empty = tensor.empty() : tensor<64x64xf32>
  %swizzle = iree_codegen.swizzle_hint %empty[#iree_codegen.rotate_rows<64, 4>] : tensor<64x64xf32>
  %slice = tensor.extract_slice %swizzle[0, 0] [16, 32] [1, 1] : tensor<64x64xf32> to tensor<16x32xf32>
  return %slice : tensor<16x32xf32>
}

// CHECK-LABEL: func.func @fold_extract_slice_of_swizzle_hint
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<16x32xf32>
//       CHECK:   %[[SWIZZLE:.+]] = iree_codegen.swizzle_hint %[[EMPTY]][#iree_codegen.rotate_rows<64, 4>] : tensor<16x32xf32>
//       CHECK:   return %[[SWIZZLE]]

// Test: tensor.extract_slice with dynamic sizes should fold correctly.
func.func @fold_extract_slice_dynamic(%size0: index, %size1: index) -> tensor<?x?xf32> {
  %empty = tensor.empty() : tensor<64x64xf32>
  %swizzle = iree_codegen.swizzle_hint %empty[#iree_codegen.xor_shuffle<128, 16>] : tensor<64x64xf32>
  %slice = tensor.extract_slice %swizzle[0, 0] [%size0, %size1] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
  return %slice : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @fold_extract_slice_dynamic
//  CHECK-SAME:   %[[SIZE0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[SIZE1:[A-Za-z0-9]+]]: index
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[SIZE0]], %[[SIZE1]]) : tensor<?x?xf32>
//       CHECK:   %[[SWIZZLE:.+]] = iree_codegen.swizzle_hint %[[EMPTY]][#iree_codegen.xor_shuffle<128, 16>] : tensor<?x?xf32>
//       CHECK:   return %[[SWIZZLE]]

// Test: tensor.expand_shape of swizzle_hint(tensor.empty) should fold
// to swizzle_hint(tensor.empty) with the expanded shape.
func.func @fold_expand_shape_of_swizzle_hint() -> tensor<4x16x64xf32> {
  %empty = tensor.empty() : tensor<64x64xf32>
  %swizzle = iree_codegen.swizzle_hint %empty[#iree_codegen.rotate_rows<64, 4>] : tensor<64x64xf32>
  %expanded = tensor.expand_shape %swizzle [[0, 1], [2]] output_shape [4, 16, 64] : tensor<64x64xf32> into tensor<4x16x64xf32>
  return %expanded : tensor<4x16x64xf32>
}

// CHECK-LABEL: func.func @fold_expand_shape_of_swizzle_hint
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<4x16x64xf32>
//       CHECK:   %[[SWIZZLE:.+]] = iree_codegen.swizzle_hint %[[EMPTY]][#iree_codegen.rotate_rows<64, 4>] : tensor<4x16x64xf32>
//       CHECK:   return %[[SWIZZLE]]

// Test: tensor.collapse_shape of swizzle_hint(tensor.empty) should fold
// to swizzle_hint(tensor.empty) with the collapsed shape.
func.func @fold_collapse_shape_of_swizzle_hint() -> tensor<64x64xf32> {
  %empty = tensor.empty() : tensor<4x16x4x16xf32>
  %swizzle = iree_codegen.swizzle_hint %empty[#iree_codegen.rotate_rows<64, 4>] : tensor<4x16x4x16xf32>
  %collapsed = tensor.collapse_shape %swizzle [[0, 1], [2, 3]] : tensor<4x16x4x16xf32> into tensor<64x64xf32>
  return %collapsed : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @fold_collapse_shape_of_swizzle_hint
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<64x64xf32>
//       CHECK:   %[[SWIZZLE:.+]] = iree_codegen.swizzle_hint %[[EMPTY]][#iree_codegen.rotate_rows<64, 4>] : tensor<64x64xf32>
//       CHECK:   return %[[SWIZZLE]]

// Negative test: extract_slice of swizzle_hint without tensor.empty source
// should NOT fold.
func.func @no_fold_extract_slice_non_empty(%arg0: tensor<64x64xf32>) -> tensor<16x32xf32> {
  %swizzle = iree_codegen.swizzle_hint %arg0[#iree_codegen.rotate_rows<64, 4>] : tensor<64x64xf32>
  %slice = tensor.extract_slice %swizzle[0, 0] [16, 32] [1, 1] : tensor<64x64xf32> to tensor<16x32xf32>
  return %slice : tensor<16x32xf32>
}

// CHECK-LABEL: func.func @no_fold_extract_slice_non_empty
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<64x64xf32>
//       CHECK:   %[[SWIZZLE:.+]] = iree_codegen.swizzle_hint %[[ARG0]][#iree_codegen.rotate_rows<64, 4>] : tensor<64x64xf32>
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[SWIZZLE]]
//       CHECK:   return %[[SLICE]]

// Negative test: expand_shape of swizzle_hint without tensor.empty source
// should NOT fold.
func.func @no_fold_expand_shape_non_empty(%arg0: tensor<64x64xf32>) -> tensor<4x16x64xf32> {
  %swizzle = iree_codegen.swizzle_hint %arg0[#iree_codegen.rotate_rows<64, 4>] : tensor<64x64xf32>
  %expanded = tensor.expand_shape %swizzle [[0, 1], [2]] output_shape [4, 16, 64] : tensor<64x64xf32> into tensor<4x16x64xf32>
  return %expanded : tensor<4x16x64xf32>
}

// CHECK-LABEL: func.func @no_fold_expand_shape_non_empty
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<64x64xf32>
//       CHECK:   %[[SWIZZLE:.+]] = iree_codegen.swizzle_hint %[[ARG0]][#iree_codegen.rotate_rows<64, 4>] : tensor<64x64xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[SWIZZLE]]
//       CHECK:   return %[[EXPANDED]]

// Test: XOR shuffle swizzle attribute is preserved through folding.
func.func @fold_xor_shuffle_swizzle() -> tensor<8x64xf32> {
  %empty = tensor.empty() : tensor<16x128xf32>
  %swizzle = iree_codegen.swizzle_hint %empty[#iree_codegen.xor_shuffle<128, 16>] : tensor<16x128xf32>
  %slice = tensor.extract_slice %swizzle[0, 0] [8, 64] [1, 1] : tensor<16x128xf32> to tensor<8x64xf32>
  return %slice : tensor<8x64xf32>
}

// CHECK-LABEL: func.func @fold_xor_shuffle_swizzle
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<8x64xf32>
//       CHECK:   %[[SWIZZLE:.+]] = iree_codegen.swizzle_hint %[[EMPTY]][#iree_codegen.xor_shuffle<128, 16>] : tensor<8x64xf32>
//       CHECK:   return %[[SWIZZLE]]

// Test: Rank-reducing extract_slice should work correctly.
func.func @fold_rank_reducing_extract_slice() -> tensor<32xf32> {
  %empty = tensor.empty() : tensor<64x64xf32>
  %swizzle = iree_codegen.swizzle_hint %empty[#iree_codegen.rotate_rows<64, 4>] : tensor<64x64xf32>
  %slice = tensor.extract_slice %swizzle[0, 0] [1, 32] [1, 1] : tensor<64x64xf32> to tensor<32xf32>
  return %slice : tensor<32xf32>
}

// CHECK-LABEL: func.func @fold_rank_reducing_extract_slice
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<32xf32>
//       CHECK:   %[[SWIZZLE:.+]] = iree_codegen.swizzle_hint %[[EMPTY]][#iree_codegen.rotate_rows<64, 4>] : tensor<32xf32>
//       CHECK:   return %[[SWIZZLE]]

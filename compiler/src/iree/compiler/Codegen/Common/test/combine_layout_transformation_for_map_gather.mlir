// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-combine-layout-transformation-for-map-gather,canonicalize,cse))" \
// RUN:   -split-input-file %s | FileCheck %s

func.func @fold_transpose(%buffer : memref<2x4x16xf32>) -> tensor<4x16x2xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<2x4x16xf32> -> tensor<2x4x16xf32>
  %init = tensor.empty() : tensor<4x16x2xf32>
  %transposed = linalg.transpose ins(%source : tensor<2x4x16xf32>) outs(%init : tensor<4x16x2xf32>) permutation = [1, 2, 0]
  return %transposed : tensor<4x16x2xf32>
}
// CHECK-LABEL: @fold_transpose
//  CHECK-SAME:   %[[BUFFER:[a-zA-Z0-9_]+]]
//       CHECK:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   %[[DEST:.+]] = tensor.empty() : tensor<4x16x2xf32>
//   CHECK-NOT:   linalg.transpose
//       CHECK:   %[[MAP_GATHER:.+]] = iree_linalg_ext.map_gather
//  CHECK-SAME:     %[[SOURCE]] into %[[DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index):
// For perm=[1,2,0], inverse_perm=[2,0,1]: output[i,j,k] = input[k,i,j], so yield [idx2,idx0,idx1].
//       CHECK:     iree_linalg_ext.yield %[[IDX2]], %[[IDX0]], %[[IDX1]],
//       CHECK:   } : tensor<2x4x16xf32> into tensor<4x16x2xf32> -> tensor<4x16x2xf32>

// -----

func.func @fold_expand_shape(%buffer : memref<8x16xf32>) -> tensor<2x4x16xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<8x16xf32> -> tensor<8x16xf32>
  %expand = tensor.expand_shape %source [[0, 1], [2]] output_shape [2, 4, 16] : tensor<8x16xf32> into tensor<2x4x16xf32>
  return %expand : tensor<2x4x16xf32>
}
// CHECK-LABEL: @fold_expand_shape
//  CHECK-SAME:   %[[BUFFER:[a-zA-Z0-9_]+]]
//       CHECK:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   %[[DEST:.+]] = tensor.empty() : tensor<2x4x16xf32>
//   CHECK-NOT:   tensor.expand_shape
//       CHECK:   %[[MAP_GATHER:.+]] = iree_linalg_ext.map_gather
//  CHECK-SAME:     %[[SOURCE]] into %[[DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index):
//       CHECK:     %[[LINEARIZE:.+]] = affine.linearize_index
//  CHECK-SAME:       [%[[IDX0]], %[[IDX1]]] by (2, 4)
//       CHECK:     iree_linalg_ext.yield %[[LINEARIZE]], %[[IDX2]],
//       CHECK:   } : tensor<8x16xf32> into tensor<2x4x16xf32> -> tensor<2x4x16xf32>

// -----

func.func @fold_collapse_shape(%buffer : memref<2x4x16xf32>) -> tensor<8x16xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<2x4x16xf32> -> tensor<2x4x16xf32>
  %collapse = tensor.collapse_shape %source [[0, 1], [2]] : tensor<2x4x16xf32> into tensor<8x16xf32>
  return %collapse : tensor<8x16xf32>
}
// CHECK-LABEL: @fold_collapse_shape
//  CHECK-SAME:   %[[BUFFER:[a-zA-Z0-9_]+]]
//       CHECK:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   %[[DEST:.+]] = tensor.empty() : tensor<8x16xf32>
//   CHECK-NOT:   tensor.collapse_shape
//       CHECK:   %[[MAP_GATHER:.+]] = iree_linalg_ext.map_gather
//  CHECK-SAME:     %[[SOURCE]] into %[[DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index):
//       CHECK:     %[[DELINEARIZE:.+]]:2 = affine.delinearize_index %[[IDX0]] into (2, 4)
//       CHECK:     iree_linalg_ext.yield %[[DELINEARIZE]]#0, %[[DELINEARIZE]]#1, %[[IDX1]],
//       CHECK:   } : tensor<2x4x16xf32> into tensor<8x16xf32> -> tensor<8x16xf32>

// -----

func.func @fold_extract_slice(%buffer : memref<64xf32>) -> tensor<16xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<64xf32> -> tensor<64xf32>
  %slice = tensor.extract_slice %source[8] [16] [1] : tensor<64xf32> to tensor<16xf32>
  return %slice : tensor<16xf32>
}
// CHECK-LABEL: @fold_extract_slice
//  CHECK-SAME:   %[[BUFFER:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
//       CHECK:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   %[[DEST:.+]] = tensor.empty() : tensor<16xf32>
//   CHECK-NOT:   tensor.extract_slice
//       CHECK:   %[[MAP_GATHER:.+]] = iree_linalg_ext.map_gather
//  CHECK-SAME:     %[[SOURCE]] into %[[DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index):
//       CHECK:     %[[NEW_IDX:.+]] = arith.addi %[[IDX0]], %[[C8]]
//       CHECK:     iree_linalg_ext.yield %[[NEW_IDX]],
//       CHECK:   } : tensor<64xf32> into tensor<16xf32> -> tensor<16xf32>

// -----

// Test folding copy into map_gather - the copy is folded and the identity
// map_gather is converted to a copy op.
func.func @fold_copy(%buffer : memref<4x16xf32>) -> tensor<4x16xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<4x16xf32> -> tensor<4x16xf32>
  %init = tensor.empty() : tensor<4x16xf32>
  %copied = linalg.copy ins(%source : tensor<4x16xf32>) outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %copied : tensor<4x16xf32>
}
// CHECK-LABEL: @fold_copy
//  CHECK-SAME:   %[[BUFFER:[a-zA-Z0-9_]+]]
//       CHECK:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   linalg.copy
//   CHECK-NOT:   iree_linalg_ext.map_gather

// -----

// Low padding is [0, 0, 0], so indices are passed through unchanged due to subi with 0.
func.func @fold_pad_with_zero_low_padding_offsets(%buffer : memref<1x50x64xf32>) -> tensor<1x64x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %source = iree_codegen.load_from_buffer %buffer : memref<1x50x64xf32> -> tensor<1x50x64xf32>
  %padded = tensor.pad %source low[0, 0, 0] high[0, 14, 0] {
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    tensor.yield %cst : f32
  } : tensor<1x50x64xf32> to tensor<1x64x64xf32>
  return %padded : tensor<1x64x64xf32>
}
// CHECK-LABEL: @fold_pad_with_zero_low_padding_offsets
//  CHECK-SAME:   %[[BUFFER:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   %[[DEST:.+]] = tensor.empty() : tensor<1x64x64xf32>
//   CHECK-NOT:   tensor.pad
//       CHECK:   %[[MAP_GATHER:.+]] = iree_linalg_ext.map_gather
//  CHECK-SAME:     %[[SOURCE]] into %[[DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index):
//       CHECK:     iree_linalg_ext.yield %[[IDX0]], %[[IDX1]], %[[IDX2]], %[[CST]] :
//       CHECK:   } : tensor<1x50x64xf32> into tensor<1x64x64xf32> -> tensor<1x64x64xf32>

// -----

func.func @fold_pad_with_non_zero_low_padding_offsets(%buffer : memref<8x16xf32>) -> tensor<10x20xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %source = iree_codegen.load_from_buffer %buffer : memref<8x16xf32> -> tensor<8x16xf32>
  %padded = tensor.pad %source low[1, 2] high[1, 2] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst : f32
  } : tensor<8x16xf32> to tensor<10x20xf32>
  return %padded : tensor<10x20xf32>
}
// CHECK-LABEL: @fold_pad_with_non_zero_low_padding_offsets
//  CHECK-SAME:   %[[BUFFER:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 1.000000e+00 : f32
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   %[[DEST:.+]] = tensor.empty() : tensor<10x20xf32>
//   CHECK-NOT:   tensor.pad
//       CHECK:   %[[MAP_GATHER:.+]] = iree_linalg_ext.map_gather
//  CHECK-SAME:     %[[SOURCE]] into %[[DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index):
//       CHECK:     %[[NEW_IDX0:.+]] = arith.subi %[[IDX0]], %[[C1]] : index
//       CHECK:     %[[NEW_IDX1:.+]] = arith.subi %[[IDX1]], %[[C2]] : index
//       CHECK:     iree_linalg_ext.yield %[[NEW_IDX0]], %[[NEW_IDX1]], %[[CST]] :
//       CHECK:   } : tensor<8x16xf32> into tensor<10x20xf32> -> tensor<10x20xf32>

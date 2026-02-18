// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-combine-source-layout-transformation,canonicalize,cse))" \
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
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.expand_shape
//   CHECK-NOT:   iree_linalg_ext.map_gather

// -----

func.func @fold_collapse_shape(%buffer : memref<2x4x16xf32>) -> tensor<8x16xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<2x4x16xf32> -> tensor<2x4x16xf32>
  %collapse = tensor.collapse_shape %source [[0, 1], [2]] : tensor<2x4x16xf32> into tensor<8x16xf32>
  return %collapse : tensor<8x16xf32>
}
// CHECK-LABEL: @fold_collapse_shape
//  CHECK-SAME:   %[[BUFFER:[a-zA-Z0-9_]+]]
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.collapse_shape
//   CHECK-NOT:   iree_linalg_ext.map_gather

// -----

func.func @fold_extract_slice(%buffer : memref<64xf32>) -> tensor<16xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<64xf32> -> tensor<64xf32>
  %slice = tensor.extract_slice %source[8] [16] [1] : tensor<64xf32> to tensor<16xf32>
  return %slice : tensor<16xf32>
}
// CHECK-LABEL: @fold_extract_slice
//  CHECK-SAME:   %[[BUFFER:[a-zA-Z0-9_]+]]
//       CHECK:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.extract_slice %[[SOURCE]][8] [16] [1]
//   CHECK-NOT:   iree_linalg_ext.map_gather

// -----

// Test folding copy into map_gather. The copy is chained with a transpose
// so the resulting map_gather is not an identity.
func.func @fold_copy_transpose(%buffer : memref<4x16xf32>) -> tensor<16x4xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<4x16xf32> -> tensor<4x16xf32>
  %init = tensor.empty() : tensor<4x16xf32>
  %copied = linalg.copy ins(%source : tensor<4x16xf32>) outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
  %init2 = tensor.empty() : tensor<16x4xf32>
  %transposed = linalg.transpose ins(%copied : tensor<4x16xf32>) outs(%init2 : tensor<16x4xf32>) permutation = [1, 0]
  return %transposed : tensor<16x4xf32>
}
// CHECK-LABEL: @fold_copy_transpose
//  CHECK-SAME:   %[[BUFFER:[a-zA-Z0-9_]+]]
//       CHECK:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   %[[DEST:.+]] = tensor.empty() : tensor<16x4xf32>
//   CHECK-NOT:   linalg.copy
//   CHECK-NOT:   linalg.transpose
//       CHECK:   %[[MAP_GATHER:.+]] = iree_linalg_ext.map_gather
//  CHECK-SAME:     %[[SOURCE]] into %[[DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index):
//       CHECK:     iree_linalg_ext.yield %[[IDX1]], %[[IDX0]],
//       CHECK:   } : tensor<4x16xf32> into tensor<16x4xf32> -> tensor<16x4xf32>

// -----

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
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.pad
//   CHECK-NOT:   iree_linalg_ext.map_gather

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
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.pad
//   CHECK-NOT:   iree_linalg_ext.map_gather

// -----

// Test that nested pads with different padding values are NOT both folded.
// The first pad gets folded, but the second pad remains because the map_gather
// already has a non-poison padding value. Folding both would incorrectly
// overwrite the first padding value.
func.func @nested_pads_different_values(%buffer : memref<8x16xf32>) -> tensor<14x24xf32> {
  %cst0 = arith.constant 0.000000e+00 : f32
  %cst1 = arith.constant 1.000000e+00 : f32
  %source = iree_codegen.load_from_buffer %buffer : memref<8x16xf32> -> tensor<8x16xf32>
  %pad0 = tensor.pad %source low[1, 2] high[1, 2] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst0 : f32
  } : tensor<8x16xf32> to tensor<10x20xf32>
  %pad1 = tensor.pad %pad0 low[2, 2] high[2, 2] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst1 : f32
  } : tensor<10x20xf32> to tensor<14x24xf32>
  return %pad1 : tensor<14x24xf32>
}
// CHECK-LABEL: @nested_pads_different_values
//       CHECK:   %[[CST0:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[CST1:.+]] = arith.constant 1.000000e+00 : f32
//       CHECK:   iree_linalg_ext.map_gather
// First pad is folded, so its padding value (0.0) is in the map_gather.
//       CHECK:     iree_linalg_ext.yield {{.*}}, %[[CST0]] :
// Second pad is NOT folded because the map_gather already has a padding value.
//       CHECK:   tensor.pad
//       CHECK:     tensor.yield %[[CST1]] : f32

// -----

func.func @fold_broadcast_generic(%buffer : memref<2x3xf32>) -> tensor<2x3x4x5xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<2x3xf32> -> tensor<2x3xf32>
  %init = tensor.empty() : tensor<2x3x4x5xf32>
  %broadcast = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%source : tensor<2x3xf32>) outs(%init : tensor<2x3x4x5xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x3x4x5xf32>
  return %broadcast : tensor<2x3x4x5xf32>
}
// CHECK-LABEL: @fold_broadcast_generic
//  CHECK-SAME:   %[[BUFFER:[a-zA-Z0-9_]+]]
//       CHECK:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   %[[DEST:.+]] = tensor.empty() : tensor<2x3x4x5xf32>
//   CHECK-NOT:   linalg.generic
//       CHECK:   %[[MAP_GATHER:.+]] = iree_linalg_ext.map_gather
//  CHECK-SAME:     %[[SOURCE]] into %[[DEST]] {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index, %[[IDX3:.+]]: index):
// Broadcast: output (d0,d1,d2,d3) reads from source at (d0,d1)
//       CHECK:     iree_linalg_ext.yield %[[IDX0]], %[[IDX1]],
//       CHECK:   } : tensor<2x3xf32> into tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>

// -----

func.func @complex_relayout_chain(%buffer : memref<8x16xf32>) -> tensor<16x8xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<8x16xf32> -> tensor<8x16xf32>
  %expanded = tensor.expand_shape %source [[0, 1], [2]] output_shape [2, 4, 16] : tensor<8x16xf32> into tensor<2x4x16xf32>
  %collapsed = tensor.collapse_shape %expanded [[0, 1], [2]] : tensor<2x4x16xf32> into tensor<8x16xf32>
  %init = tensor.empty() : tensor<16x8xf32>
  %transposed = linalg.transpose ins(%collapsed : tensor<8x16xf32>) outs(%init : tensor<16x8xf32>) permutation = [1, 0]
  return %transposed : tensor<16x8xf32>
}
// CHECK-LABEL: @complex_relayout_chain
//  CHECK-SAME:   %[[BUFFER:[a-zA-Z0-9_]+]]
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.empty() : tensor<16x8xf32>
//   CHECK-NOT:   tensor.expand_shape
//   CHECK-NOT:   tensor.collapse_shape
//   CHECK-NOT:   linalg.transpose
//       CHECK:   iree_linalg_ext.map_gather {{.*}} into {{.*}} {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index):
//  CHECK-NEXT:     iree_linalg_ext.yield %[[IDX1]], %[[IDX0]], {{.*}} : index, index, f32
//       CHECK:   } : tensor<8x16xf32> into tensor<16x8xf32> -> tensor<16x8xf32>

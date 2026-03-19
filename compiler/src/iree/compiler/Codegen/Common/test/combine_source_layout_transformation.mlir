// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-combine-source-layout-transformation,canonicalize,cse))" \
// RUN:   -split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-combine-source-layout-transformation{test-combine-non-complex-chains=true},canonicalize,cse))" \
// RUN:   -split-input-file %s | FileCheck %s --check-prefix=FOLD

func.func @transpose(%buffer : memref<2x4x16xf32>) -> tensor<4x16x2xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<2x4x16xf32> -> tensor<2x4x16xf32>
  %init = tensor.empty() : tensor<4x16x2xf32>
  %transposed = linalg.transpose ins(%source : tensor<2x4x16xf32>) outs(%init : tensor<4x16x2xf32>) permutation = [1, 2, 0]
  return %transposed : tensor<4x16x2xf32>
}
// CHECK-LABEL: @transpose
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   linalg.transpose
//   CHECK-NOT:   iree_linalg_ext.map_load
// FOLD-LABEL: @transpose
//  FOLD-SAME:   %[[BUFFER:.+]]:
//       FOLD:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       FOLD:   %[[DEST:.+]] = tensor.empty() : tensor<4x16x2xf32>
//   FOLD-NOT:   linalg.transpose
//       FOLD:   %[[MAP_LOAD:.+]] = iree_linalg_ext.map_load
//  FOLD-SAME:     %[[SOURCE]] into %[[DEST]] {
//  FOLD-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index):
//       FOLD:     iree_linalg_ext.yield %[[IDX2]], %[[IDX0]], %[[IDX1]], {{.*}} : index, index, index, f32
//       FOLD:   } : tensor<2x4x16xf32> into tensor<4x16x2xf32> -> tensor<4x16x2xf32>

// -----

func.func @expand_shape(%buffer : memref<8x16xf32>) -> tensor<2x4x16xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<8x16xf32> -> tensor<8x16xf32>
  %expand = tensor.expand_shape %source [[0, 1], [2]] output_shape [2, 4, 16] : tensor<8x16xf32> into tensor<2x4x16xf32>
  return %expand : tensor<2x4x16xf32>
}
// CHECK-LABEL: @expand_shape
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.expand_shape
//   CHECK-NOT:   iree_linalg_ext.map_load
// FOLD-LABEL: @expand_shape
//  FOLD-SAME:   %[[BUFFER:.+]]:
//       FOLD:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       FOLD:   %[[DEST:.+]] = tensor.empty() : tensor<2x4x16xf32>
//   FOLD-NOT:   tensor.expand_shape
//       FOLD:   %[[MAP_LOAD:.+]] = iree_linalg_ext.map_load
//  FOLD-SAME:     %[[SOURCE]] into %[[DEST]] {
//  FOLD-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index):
//       FOLD:     %[[LINEARIZE:.+]] = affine.linearize_index disjoint [%[[IDX0]], %[[IDX1]]] by (2, 4) : index
//       FOLD:     iree_linalg_ext.yield %[[LINEARIZE]], %[[IDX2]], {{.*}} : index, index, f32
//       FOLD:   } : tensor<8x16xf32> into tensor<2x4x16xf32> -> tensor<2x4x16xf32>

// -----

func.func @collapse_shape(%buffer : memref<2x4x16xf32>) -> tensor<8x16xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<2x4x16xf32> -> tensor<2x4x16xf32>
  %collapse = tensor.collapse_shape %source [[0, 1], [2]] : tensor<2x4x16xf32> into tensor<8x16xf32>
  return %collapse : tensor<8x16xf32>
}
// CHECK-LABEL: @collapse_shape
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.collapse_shape
//   CHECK-NOT:   iree_linalg_ext.map_load
// FOLD-LABEL: @collapse_shape
//  FOLD-SAME:   %[[BUFFER:.+]]:
//       FOLD:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       FOLD:   %[[DEST:.+]] = tensor.empty() : tensor<8x16xf32>
//   FOLD-NOT:   tensor.collapse_shape
//       FOLD:   %[[MAP_LOAD:.+]] = iree_linalg_ext.map_load
//  FOLD-SAME:     %[[SOURCE]] into %[[DEST]] {
//  FOLD-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index):
//       FOLD:     %[[DELINEARIZE:.+]]:2 = affine.delinearize_index %[[IDX0]] into (2, 4) : index, index
//       FOLD:     iree_linalg_ext.yield %[[DELINEARIZE]]#0, %[[DELINEARIZE]]#1, %[[IDX1]], {{.*}} : index, index, index, f32
//       FOLD:   } : tensor<2x4x16xf32> into tensor<8x16xf32> -> tensor<8x16xf32>

// -----

// Verify that collapse_shape to rank-0 (scalar) does not fold into map_load,
// because map_load requires non-zero rank output.
func.func @no_fold_rank0_collapse_shape(%buffer : memref<1x1x1xf32>) -> tensor<f32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<1x1x1xf32> -> tensor<1x1x1xf32>
  %collapse = tensor.collapse_shape %source [] : tensor<1x1x1xf32> into tensor<f32>
  return %collapse : tensor<f32>
}
// CHECK-LABEL: @no_fold_rank0_collapse_shape
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.collapse_shape
//   CHECK-NOT:   iree_linalg_ext.map_load
// FOLD-LABEL: @no_fold_rank0_collapse_shape
//  FOLD-SAME:   %[[BUFFER:.+]]:
//       FOLD:   iree_codegen.load_from_buffer %[[BUFFER]]
//       FOLD:   tensor.collapse_shape
//   FOLD-NOT:   iree_linalg_ext.map_load

// -----

// Verify that expand_shape from rank-0 (scalar) does not fold into map_load,
// because the delinearization would produce no indices.
func.func @no_fold_rank0_expand_shape(%buffer : memref<f32>) -> tensor<1x1x1xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<f32> -> tensor<f32>
  %expand = tensor.expand_shape %source [] output_shape [1, 1, 1] : tensor<f32> into tensor<1x1x1xf32>
  return %expand : tensor<1x1x1xf32>
}
// CHECK-LABEL: @no_fold_rank0_expand_shape
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.expand_shape
//   CHECK-NOT:   iree_linalg_ext.map_load
// FOLD-LABEL: @no_fold_rank0_expand_shape
//  FOLD-SAME:   %[[BUFFER:.+]]:
//       FOLD:   iree_codegen.load_from_buffer %[[BUFFER]]
//       FOLD:   tensor.expand_shape
//   FOLD-NOT:   iree_linalg_ext.map_load

// -----

func.func @extract_slice(%buffer : memref<64xf32>) -> tensor<16xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<64xf32> -> tensor<64xf32>
  %slice = tensor.extract_slice %source[8] [16] [1] : tensor<64xf32> to tensor<16xf32>
  return %slice : tensor<16xf32>
}
// CHECK-LABEL: @extract_slice
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.extract_slice %[[SOURCE]][8] [16] [1]
//   CHECK-NOT:   iree_linalg_ext.map_load
// FOLD-LABEL: @extract_slice
//  FOLD-SAME:   %[[BUFFER:.+]]:
//   FOLD-DAG:   %[[C8:.+]] = arith.constant 8 : index
//       FOLD:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       FOLD:   %[[DEST:.+]] = tensor.empty() : tensor<16xf32>
//   FOLD-NOT:   tensor.extract_slice
//       FOLD:   %[[MAP_LOAD:.+]] = iree_linalg_ext.map_load
//  FOLD-SAME:     %[[SOURCE]] into %[[DEST]] {
//  FOLD-NEXT:   ^bb0(%[[IDX0:.+]]: index):
//       FOLD:     %[[NEW_IDX:.+]] = arith.addi %[[IDX0]], %[[C8]] overflow<nsw> : index
//       FOLD:     iree_linalg_ext.yield %[[NEW_IDX]], {{.*}} : index, f32
//       FOLD:   } : tensor<64xf32> into tensor<16xf32> -> tensor<16xf32>

// -----

func.func @copy_transpose(%buffer : memref<4x16xf32>) -> tensor<16x4xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<4x16xf32> -> tensor<4x16xf32>
  %init = tensor.empty() : tensor<4x16xf32>
  %copied = linalg.copy ins(%source : tensor<4x16xf32>) outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
  %init2 = tensor.empty() : tensor<16x4xf32>
  %transposed = linalg.transpose ins(%copied : tensor<4x16xf32>) outs(%init2 : tensor<16x4xf32>) permutation = [1, 0]
  return %transposed : tensor<16x4xf32>
}
// CHECK-LABEL: @copy_transpose
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   linalg.copy
//       CHECK:   linalg.transpose
//   CHECK-NOT:   iree_linalg_ext.map_load
// FOLD-LABEL: @copy_transpose
//  FOLD-SAME:   %[[BUFFER:.+]]:
//       FOLD:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       FOLD:   %[[DEST:.+]] = tensor.empty() : tensor<16x4xf32>
//   FOLD-NOT:   linalg.copy
//   FOLD-NOT:   linalg.transpose
//       FOLD:   %[[MAP_LOAD:.+]] = iree_linalg_ext.map_load
//  FOLD-SAME:     %[[SOURCE]] into %[[DEST]] {
//  FOLD-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index):
//       FOLD:     iree_linalg_ext.yield %[[IDX1]], %[[IDX0]], {{.*}} : index, index, f32
//       FOLD:   } : tensor<4x16xf32> into tensor<16x4xf32> -> tensor<16x4xf32>

// -----

func.func @pad_zero_low(%buffer : memref<1x50x64xf32>) -> tensor<1x64x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %source = iree_codegen.load_from_buffer %buffer : memref<1x50x64xf32> -> tensor<1x50x64xf32>
  %padded = tensor.pad %source low[0, 0, 0] high[0, 14, 0] {
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    tensor.yield %cst : f32
  } : tensor<1x50x64xf32> to tensor<1x64x64xf32>
  return %padded : tensor<1x64x64xf32>
}
// CHECK-LABEL: @pad_zero_low
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.pad
//   CHECK-NOT:   iree_linalg_ext.map_load
// FOLD-LABEL: @pad_zero_low
//  FOLD-SAME:   %[[BUFFER:.+]]:
//   FOLD-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       FOLD:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       FOLD:   %[[DEST:.+]] = tensor.empty() : tensor<1x64x64xf32>
//   FOLD-NOT:   tensor.pad
//       FOLD:   %[[MAP_LOAD:.+]] = iree_linalg_ext.map_load
//  FOLD-SAME:     %[[SOURCE]] into %[[DEST]] {
//  FOLD-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index):
//       FOLD:     iree_linalg_ext.yield %[[IDX0]], %[[IDX1]], %[[IDX2]], %[[CST]] : index, index, index, f32
//       FOLD:   } : tensor<1x50x64xf32> into tensor<1x64x64xf32> -> tensor<1x64x64xf32>

// -----

func.func @pad_non_zero_low(%buffer : memref<8x16xf32>) -> tensor<10x20xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %source = iree_codegen.load_from_buffer %buffer : memref<8x16xf32> -> tensor<8x16xf32>
  %padded = tensor.pad %source low[1, 2] high[1, 2] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst : f32
  } : tensor<8x16xf32> to tensor<10x20xf32>
  return %padded : tensor<10x20xf32>
}
// CHECK-LABEL: @pad_non_zero_low
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.pad
//   CHECK-NOT:   iree_linalg_ext.map_load
// FOLD-LABEL: @pad_non_zero_low
//  FOLD-SAME:   %[[BUFFER:.+]]:
//   FOLD-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   FOLD-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   FOLD-DAG:   %[[CST:.+]] = arith.constant 1.000000e+00 : f32
//       FOLD:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       FOLD:   %[[DEST:.+]] = tensor.empty() : tensor<10x20xf32>
//   FOLD-NOT:   tensor.pad
//       FOLD:   %[[MAP_LOAD:.+]] = iree_linalg_ext.map_load
//  FOLD-SAME:     %[[SOURCE]] into %[[DEST]] {
//  FOLD-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index):
//       FOLD:     %[[NEW_IDX0:.+]] = arith.subi %[[IDX0]], %[[C1]] overflow<nsw> : index
//       FOLD:     %[[NEW_IDX1:.+]] = arith.subi %[[IDX1]], %[[C2]] overflow<nsw> : index
//       FOLD:     iree_linalg_ext.yield %[[NEW_IDX0]], %[[NEW_IDX1]], %[[CST]] : index, index, f32
//       FOLD:   } : tensor<8x16xf32> into tensor<10x20xf32> -> tensor<10x20xf32>

// -----

// Chain with pad -> reshape -> pad. Reshape makes it complex, so first pad is folded
// into map_load. Second pad is NOT folded because map_load already has a padding value.
func.func @nested_pads_different_values(%buffer : memref<8x16xf32>) -> tensor<2x7x24xf32> {
  %cst0 = arith.constant 0.000000e+00 : f32
  %cst1 = arith.constant 1.000000e+00 : f32
  %source = iree_codegen.load_from_buffer %buffer : memref<8x16xf32> -> tensor<8x16xf32>
  %pad0 = tensor.pad %source low[1, 2] high[1, 2] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst0 : f32
  } : tensor<8x16xf32> to tensor<10x20xf32>
  %expanded = tensor.expand_shape %pad0 [[0, 1], [2]] output_shape [2, 5, 20] : tensor<10x20xf32> into tensor<2x5x20xf32>
  %pad1 = tensor.pad %expanded low[0, 1, 2] high[0, 1, 2] {
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    tensor.yield %cst1 : f32
  } : tensor<2x5x20xf32> to tensor<2x7x24xf32>
  return %pad1 : tensor<2x7x24xf32>
}
// CHECK-LABEL: @nested_pads_different_values
//       CHECK:   %[[CST0:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[CST1:.+]] = arith.constant 1.000000e+00 : f32
//       CHECK:   iree_linalg_ext.map_load
// First pad is folded, so its padding value (0.0) is in the map_load.
//       CHECK:     iree_linalg_ext.yield {{.*}}, %[[CST0]] :
// Second pad is NOT folded because the map_load already has a padding value.
//       CHECK:   tensor.pad
//       CHECK:     tensor.yield %[[CST1]] : f32
// FOLD-LABEL: @nested_pads_different_values
//  FOLD-SAME:   %[[BUFFER:.+]]:
// With test-combine-non-complex-chains, first pad folds; second pad still remains.
//   FOLD-DAG:   %[[CST0:.+]] = arith.constant 0.000000e+00 : f32
//       FOLD:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       FOLD:   %[[DEST:.+]] = tensor.empty() : tensor<2x5x20xf32>
//       FOLD:   %[[MAP_LOAD:.+]] = iree_linalg_ext.map_load
//  FOLD-SAME:     %[[SOURCE]] into %[[DEST]] {
//  FOLD-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index):
//       FOLD:     {{.*}} = affine.linearize_index disjoint [%[[IDX0]], %[[IDX1]]] by (2, 5) : index
//       FOLD:     {{.*}} = arith.subi {{.*}} overflow<nsw> : index
//       FOLD:     iree_linalg_ext.yield {{.*}}, %[[CST0]] : index, index, f32
//       FOLD:   } : tensor<8x16xf32> into tensor<2x5x20xf32> -> tensor<2x5x20xf32>
//       FOLD:   tensor.pad %[[MAP_LOAD]]

// -----

func.func @broadcast_generic(%buffer : memref<2x3xf32>) -> tensor<2x3x4x5xf32> {
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
// CHECK-LABEL: @broadcast_generic
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   linalg.generic
//   CHECK-NOT:   iree_linalg_ext.map_load
// FOLD-LABEL: @broadcast_generic
//  FOLD-SAME:   %[[BUFFER:.+]]:
//       FOLD:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       FOLD:   %[[DEST:.+]] = tensor.empty() : tensor<2x3x4x5xf32>
//   FOLD-NOT:   linalg.generic
//       FOLD:   %[[MAP_LOAD:.+]] = iree_linalg_ext.map_load
//  FOLD-SAME:     %[[SOURCE]] into %[[DEST]] {
//  FOLD-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index, %[[IDX3:.+]]: index):
//       FOLD:     iree_linalg_ext.yield %[[IDX0]], %[[IDX1]], {{.*}} : index, index, f32
//       FOLD:   } : tensor<2x3xf32> into tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>

// -----

func.func @broadcast_named(%buffer : memref<2x3xf32>) -> tensor<2x3x4x5xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<2x3xf32> -> tensor<2x3xf32>
  %init = tensor.empty() : tensor<2x3x4x5xf32>
  %broadcast = linalg.broadcast ins(%source : tensor<2x3xf32>) outs(%init : tensor<2x3x4x5xf32>) dimensions = [2, 3]
  return %broadcast : tensor<2x3x4x5xf32>
}
// CHECK-LABEL: @broadcast_named
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   linalg.broadcast
//   CHECK-NOT:   iree_linalg_ext.map_load
// FOLD-LABEL: @broadcast_named
//  FOLD-SAME:   %[[BUFFER:.+]]:
//       FOLD:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       FOLD:   %[[DEST:.+]] = tensor.empty() : tensor<2x3x4x5xf32>
//   FOLD-NOT:   linalg.broadcast
//       FOLD:   %[[MAP_LOAD:.+]] = iree_linalg_ext.map_load
//  FOLD-SAME:     %[[SOURCE]] into %[[DEST]] {
//  FOLD-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index, %[[IDX3:.+]]: index):
//       FOLD:     iree_linalg_ext.yield %[[IDX0]], %[[IDX1]], {{.*}} : index, index, f32
//       FOLD:   } : tensor<2x3xf32> into tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>

// -----

func.func @complex_relayout_chain(%buffer : memref<8x16xf32>) -> tensor<16x8xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<8x16xf32> -> tensor<8x16xf32>
  %expanded = tensor.expand_shape %source [[0, 1], [2]] output_shape [2, 4, 16] : tensor<8x16xf32> into tensor<2x4x16xf32>
  %init_t = tensor.empty() : tensor<16x2x4xf32>
  %transposed = linalg.transpose ins(%expanded : tensor<2x4x16xf32>) outs(%init_t : tensor<16x2x4xf32>) permutation = [2, 0, 1]
  %collapsed = tensor.collapse_shape %transposed [[0], [1, 2]] : tensor<16x2x4xf32> into tensor<16x8xf32>
  return %collapsed : tensor<16x8xf32>
}
// CHECK-LABEL: @complex_relayout_chain
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.empty() : tensor<16x8xf32>
//   CHECK-NOT:   tensor.expand_shape
//   CHECK-NOT:   linalg.transpose
//   CHECK-NOT:   tensor.collapse_shape
//       CHECK:   iree_linalg_ext.map_load {{.*}} into {{.*}} {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index):
//       CHECK:     iree_linalg_ext.yield %[[IDX1]], %[[IDX0]], {{.*}} : index, index, f32
//       CHECK:   } : tensor<8x16xf32> into tensor<16x8xf32> -> tensor<16x8xf32>
// FOLD-LABEL: @complex_relayout_chain
//  FOLD-SAME:   %[[BUFFER:.+]]:
//       FOLD:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       FOLD:   %[[DEST:.+]] = tensor.empty() : tensor<16x8xf32>
//       FOLD:   %[[MAP_LOAD:.+]] = iree_linalg_ext.map_load
//  FOLD-SAME:     %[[SOURCE]] into %[[DEST]] {
//  FOLD-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index):
//       FOLD:     iree_linalg_ext.yield %[[IDX1]], %[[IDX0]], {{.*}} : index, index, f32
//       FOLD:   } : tensor<8x16xf32> into tensor<16x8xf32> -> tensor<16x8xf32>

// -----

// Chain with reshape (expand_shape) + non-extract_slice (transpose) - complex chain,
// so map_load is introduced and the chain is folded.
func.func @complex_chain_reshape_and_transpose(%buffer : memref<4x8xf32>) -> tensor<8x2x2xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<4x8xf32> -> tensor<4x8xf32>
  %expanded = tensor.expand_shape %source [[0, 1], [2]] output_shape [2, 2, 8] : tensor<4x8xf32> into tensor<2x2x8xf32>
  %init = tensor.empty() : tensor<8x2x2xf32>
  %transposed = linalg.transpose ins(%expanded : tensor<2x2x8xf32>) outs(%init : tensor<8x2x2xf32>) permutation = [2, 0, 1]
  return %transposed : tensor<8x2x2xf32>
}
// CHECK-LABEL: @complex_chain_reshape_and_transpose
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.empty() : tensor<8x2x2xf32>
//   CHECK-NOT:   tensor.expand_shape
//   CHECK-NOT:   linalg.transpose
//       CHECK:   iree_linalg_ext.map_load {{.*}} into {{.*}} {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index):
//       CHECK:     %[[LINEAR:.+]] = affine.linearize_index disjoint [%[[IDX1]], %[[IDX2]]] by (2, 2) : index
//       CHECK:     iree_linalg_ext.yield %[[LINEAR]], %[[IDX0]], %0 : index, index, f32
//       CHECK:   } : tensor<4x8xf32> into tensor<8x2x2xf32> -> tensor<8x2x2xf32>
// FOLD-LABEL: @complex_chain_reshape_and_transpose
//  FOLD-SAME:   %[[BUFFER:.+]]:
//       FOLD:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       FOLD:   %[[DEST:.+]] = tensor.empty() : tensor<8x2x2xf32>
//       FOLD:   %[[MAP_LOAD:.+]] = iree_linalg_ext.map_load
//  FOLD-SAME:     %[[SOURCE]] into %[[DEST]] {
//  FOLD-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index):
//       FOLD:     %[[LINEAR:.+]] = affine.linearize_index disjoint [%[[IDX1]], %[[IDX2]]] by (2, 2) : index
//       FOLD:     iree_linalg_ext.yield %[[LINEAR]], %[[IDX0]], {{.*}} : index, index, f32
//       FOLD:   } : tensor<4x8xf32> into tensor<8x2x2xf32> -> tensor<8x2x2xf32>

// -----

// Chain with a complex relayout op chain feeding into a tensor.insert_slice.
func.func @complex_chain_into_insert_slice(%buffer : memref<4x8xf32>) -> tensor<16x4x4xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<4x8xf32> -> tensor<4x8xf32>
  %expanded = tensor.expand_shape %source [[0, 1], [2]] output_shape [2, 2, 8] : tensor<4x8xf32> into tensor<2x2x8xf32>
  %init = tensor.empty() : tensor<8x2x2xf32>
  %transposed = linalg.transpose ins(%expanded : tensor<2x2x8xf32>) outs(%init : tensor<8x2x2xf32>) permutation = [2, 0, 1]
  %dest = tensor.empty() : tensor<16x4x4xf32>
  %result = tensor.insert_slice %transposed into %dest[0, 0, 0] [8, 2, 2] [1, 1, 1] : tensor<8x2x2xf32> into tensor<16x4x4xf32>
  return %result : tensor<16x4x4xf32>
}
// CHECK-LABEL: @complex_chain_into_insert_slice
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.expand_shape
//       CHECK:   linalg.transpose
//   CHECK-NOT:   iree_linalg_ext.map_load
//       CHECK:   tensor.insert_slice
// FOLD-LABEL: @complex_chain_into_insert_slice
//  FOLD-SAME:   %[[BUFFER:.+]]:
//       FOLD:   iree_codegen.load_from_buffer
//       FOLD:   iree_linalg_ext.map_load
//       FOLD:   } : tensor<4x8xf32> into tensor<8x2x2xf32> -> tensor<8x2x2xf32>
//       FOLD:   tensor.insert_slice

// -----

// Chain broadcast -> pad -> expand_shape folds into map_load, but copy doesn't
// because it uses expand_shape's result (later map_load) as outs (operand 1).
func.func @fold_broadcast_pad_expand_shape(%buffer : memref<2x64xf32>, %batch : index) -> tensor<1x4x16x4x2x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %source = iree_codegen.load_from_buffer %buffer : memref<2x64xf32> -> tensor<2x64xf32>
  %extracted = tensor.extract_slice %source[%batch, 0] [1, 64] [1, 1] : tensor<2x64xf32> to tensor<1x64xf32>
  %init = tensor.empty() : tensor<1x64x4x28xf32>
  %broadcast = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%extracted : tensor<1x64xf32>) outs(%init : tensor<1x64x4x28xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x64x4x28xf32>
  %padded = tensor.pad %broadcast low[0, 0, 0, 0] high[0, 0, 0, 4] {
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
    tensor.yield %cst : f32
  } : tensor<1x64x4x28xf32> to tensor<1x64x4x32xf32>
  %expanded = tensor.expand_shape %padded [[0], [1, 2], [3], [4, 5]] output_shape [1, 4, 16, 4, 2, 16] : tensor<1x64x4x32xf32> into tensor<1x4x16x4x2x16xf32>
  %copy_dest = tensor.empty() : tensor<1x4x16x4x2x16xf32>
  %result = linalg.copy ins(%copy_dest : tensor<1x4x16x4x2x16xf32>) outs(%expanded : tensor<1x4x16x4x2x16xf32>) -> tensor<1x4x16x4x2x16xf32>
  return %result : tensor<1x4x16x4x2x16xf32>
}
// CHECK-LABEL: @fold_broadcast_pad_expand_shape
//  CHECK-SAME:   %[[BUFFER:.+]]: memref<2x64xf32>, %[[BATCH:.+]]: index
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.empty() : tensor<1x4x16x4x2x16xf32>
//   CHECK-NOT:   tensor.extract_slice
//   CHECK-NOT:   linalg.generic
//   CHECK-NOT:   tensor.pad
//   CHECK-NOT:   tensor.expand_shape
//       CHECK:   %[[MAP_LOAD:.+]] = iree_linalg_ext.map_load {{.*}} into {{.*}} {
//  CHECK-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index, %[[IDX3:.+]]: index, %[[IDX4:.+]]: index, %[[IDX5:.+]]: index):
//       CHECK:     {{.*}} = affine.linearize_index disjoint [%[[IDX1]], %[[IDX2]], %[[IDX3]], %[[IDX4]], %[[IDX5]]] by (4, 16, 4, 2, 16) : index
//       CHECK:     {{.*}} = affine.delinearize_index {{.*}} into (64, 4, 32) : index, index, index
//       CHECK:     iree_linalg_ext.yield %[[BATCH]], {{.*}}, {{.*}} : index, index, f32
//       CHECK:   } : tensor<2x64xf32> into tensor<1x4x16x4x2x16xf32> -> tensor<1x4x16x4x2x16xf32>
//       CHECK:   linalg.copy ins({{.*}}) outs(%[[MAP_LOAD]]
// FOLD-LABEL: @fold_broadcast_pad_expand_shape
//  FOLD-SAME:   %[[BUFFER:.+]]: memref<2x64xf32>, %[[BATCH:.+]]: index
//   FOLD-DAG:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       FOLD:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       FOLD:   %[[DEST:.+]] = tensor.empty() : tensor<1x4x16x4x2x16xf32>
//       FOLD:   %[[MAP_LOAD:.+]] = iree_linalg_ext.map_load
//  FOLD-SAME:     %[[SOURCE]] into %[[DEST]] {
//  FOLD-NEXT:   ^bb0(%[[IDX0:.+]]: index, %[[IDX1:.+]]: index, %[[IDX2:.+]]: index, %[[IDX3:.+]]: index, %[[IDX4:.+]]: index, %[[IDX5:.+]]: index):
//       FOLD:     {{.*}} = affine.linearize_index disjoint [%[[IDX1]], %[[IDX2]], %[[IDX3]], %[[IDX4]], %[[IDX5]]] by (4, 16, 4, 2, 16) : index
//       FOLD:     {{.*}}:3 = affine.delinearize_index {{.*}} into (64, 4, 32) : index, index, index
//       FOLD:     iree_linalg_ext.yield %[[BATCH]], {{.*}}, %[[CST]] : index, index, f32
//       FOLD:   } : tensor<2x64xf32> into tensor<1x4x16x4x2x16xf32> -> tensor<1x4x16x4x2x16xf32>
//       FOLD:   linalg.copy

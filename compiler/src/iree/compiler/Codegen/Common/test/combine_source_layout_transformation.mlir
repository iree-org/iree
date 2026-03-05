// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-combine-source-layout-transformation,canonicalize,cse))" \
// RUN:   -split-input-file %s | FileCheck %s

func.func @no_fold_transpose(%buffer : memref<2x4x16xf32>) -> tensor<4x16x2xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<2x4x16xf32> -> tensor<2x4x16xf32>
  %init = tensor.empty() : tensor<4x16x2xf32>
  %transposed = linalg.transpose ins(%source : tensor<2x4x16xf32>) outs(%init : tensor<4x16x2xf32>) permutation = [1, 2, 0]
  return %transposed : tensor<4x16x2xf32>
}
// CHECK-LABEL: @no_fold_transpose
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   linalg.transpose
//   CHECK-NOT:   iree_linalg_ext.map_load

// -----

func.func @no_fold_expand_shape(%buffer : memref<8x16xf32>) -> tensor<2x4x16xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<8x16xf32> -> tensor<8x16xf32>
  %expand = tensor.expand_shape %source [[0, 1], [2]] output_shape [2, 4, 16] : tensor<8x16xf32> into tensor<2x4x16xf32>
  return %expand : tensor<2x4x16xf32>
}
// CHECK-LABEL: @no_fold_expand_shape
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.expand_shape
//   CHECK-NOT:   iree_linalg_ext.map_load

// -----

func.func @no_fold_collapse_shape(%buffer : memref<2x4x16xf32>) -> tensor<8x16xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<2x4x16xf32> -> tensor<2x4x16xf32>
  %collapse = tensor.collapse_shape %source [[0, 1], [2]] : tensor<2x4x16xf32> into tensor<8x16xf32>
  return %collapse : tensor<8x16xf32>
}
// CHECK-LABEL: @no_fold_collapse_shape
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.collapse_shape
//   CHECK-NOT:   iree_linalg_ext.map_load

// -----

// Verify that collapse_shape to rank-0 (scalar) does not fold into map_load,
// because map_load requires non-zero rank output.
func.func @no_fold_rank0_collapse_shape(%buffer : memref<1x1x1xf32>) -> tensor<f32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<1x1x1xf32> -> tensor<1x1x1xf32>
  %collapse = tensor.collapse_shape %source [] : tensor<1x1x1xf32> into tensor<f32>
  return %collapse : tensor<f32>
}
// CHECK-LABEL: @no_fold_rank0_collapse_shape
//       CHECK:   tensor.collapse_shape
//   CHECK-NOT:   iree_linalg_ext.map_load

// -----

// Verify that expand_shape from rank-0 (scalar) does not fold into map_load,
// because the delinearization would produce no indices.
func.func @no_fold_rank0_expand_shape(%buffer : memref<f32>) -> tensor<1x1x1xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<f32> -> tensor<f32>
  %expand = tensor.expand_shape %source [] output_shape [1, 1, 1] : tensor<f32> into tensor<1x1x1xf32>
  return %expand : tensor<1x1x1xf32>
}
// CHECK-LABEL: @no_fold_rank0_expand_shape
//       CHECK:   tensor.expand_shape
//   CHECK-NOT:   iree_linalg_ext.map_load

// -----

func.func @fold_extract_slice(%buffer : memref<64xf32>) -> tensor<16xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<64xf32> -> tensor<64xf32>
  %slice = tensor.extract_slice %source[8] [16] [1] : tensor<64xf32> to tensor<16xf32>
  return %slice : tensor<16xf32>
}
// CHECK-LABEL: @no_fold_extract_slice
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   %[[SOURCE:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.extract_slice %[[SOURCE]][8] [16] [1]
//   CHECK-NOT:   iree_linalg_ext.map_load

// -----

func.func @no_fold_copy_transpose(%buffer : memref<4x16xf32>) -> tensor<16x4xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<4x16xf32> -> tensor<4x16xf32>
  %init = tensor.empty() : tensor<4x16xf32>
  %copied = linalg.copy ins(%source : tensor<4x16xf32>) outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
  %init2 = tensor.empty() : tensor<16x4xf32>
  %transposed = linalg.transpose ins(%copied : tensor<4x16xf32>) outs(%init2 : tensor<16x4xf32>) permutation = [1, 0]
  return %transposed : tensor<16x4xf32>
}
// CHECK-LABEL: @no_fold_copy_transpose
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   linalg.copy
//       CHECK:   linalg.transpose
//   CHECK-NOT:   iree_linalg_ext.map_load

// -----

func.func @no_fold_pad_with_zero_low_padding_offsets(%buffer : memref<1x50x64xf32>) -> tensor<1x64x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %source = iree_codegen.load_from_buffer %buffer : memref<1x50x64xf32> -> tensor<1x50x64xf32>
  %padded = tensor.pad %source low[0, 0, 0] high[0, 14, 0] {
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    tensor.yield %cst : f32
  } : tensor<1x50x64xf32> to tensor<1x64x64xf32>
  return %padded : tensor<1x64x64xf32>
}
// CHECK-LABEL: @no_fold_pad_with_zero_low_padding_offsets
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.pad
//   CHECK-NOT:   iree_linalg_ext.map_load

// -----

func.func @no_fold_pad_with_non_zero_low_padding_offsets(%buffer : memref<8x16xf32>) -> tensor<10x20xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %source = iree_codegen.load_from_buffer %buffer : memref<8x16xf32> -> tensor<8x16xf32>
  %padded = tensor.pad %source low[1, 2] high[1, 2] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst : f32
  } : tensor<8x16xf32> to tensor<10x20xf32>
  return %padded : tensor<10x20xf32>
}
// CHECK-LABEL: @no_fold_pad_with_non_zero_low_padding_offsets
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   tensor.pad
//   CHECK-NOT:   iree_linalg_ext.map_load

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

// -----

func.func @no_fold_broadcast_generic(%buffer : memref<2x3xf32>) -> tensor<2x3x4x5xf32> {
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
// CHECK-LABEL: @no_fold_broadcast_generic
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   linalg.generic
//   CHECK-NOT:   iree_linalg_ext.map_load

// -----

func.func @no_fold_broadcast_named(%buffer : memref<2x3xf32>) -> tensor<2x3x4x5xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<2x3xf32> -> tensor<2x3xf32>
  %init = tensor.empty() : tensor<2x3x4x5xf32>
  %broadcast = linalg.broadcast ins(%source : tensor<2x3xf32>) outs(%init : tensor<2x3x4x5xf32>) dimensions = [2, 3]
  return %broadcast : tensor<2x3x4x5xf32>
}
// CHECK-LABEL: @no_fold_broadcast_named
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   linalg.broadcast
//   CHECK-NOT:   iree_linalg_ext.map_load

// -----

// Due to applyPatternsGreedily collapse(expand(x)) gets folded to x, so we end up
// with load->transpose before our patterns see the chain. And since single transpose
// is not complex - so no map_load op is inserted.
func.func @complex_relayout_chain(%buffer : memref<8x16xf32>) -> tensor<16x8xf32> {
  %source = iree_codegen.load_from_buffer %buffer : memref<8x16xf32> -> tensor<8x16xf32>
  %expanded = tensor.expand_shape %source [[0, 1], [2]] output_shape [2, 4, 16] : tensor<8x16xf32> into tensor<2x4x16xf32>
  %collapsed = tensor.collapse_shape %expanded [[0, 1], [2]] : tensor<2x4x16xf32> into tensor<8x16xf32>
  %init = tensor.empty() : tensor<16x8xf32>
  %transposed = linalg.transpose ins(%collapsed : tensor<8x16xf32>) outs(%init : tensor<16x8xf32>) permutation = [1, 0]
  return %transposed : tensor<16x8xf32>
}
// CHECK-LABEL: @complex_relayout_chain
//  CHECK-SAME:   %[[BUFFER:.+]]:
//       CHECK:   %[[LOAD:.+]] = iree_codegen.load_from_buffer %[[BUFFER]]
//       CHECK:   linalg.transpose ins(%[[LOAD]]
//   CHECK-NOT:   iree_linalg_ext.map_load

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

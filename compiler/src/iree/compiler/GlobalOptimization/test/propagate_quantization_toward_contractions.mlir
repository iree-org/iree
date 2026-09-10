// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-global-opt-propagate-quantization-toward-contractions))" %s | FileCheck %s

#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#scalar3 = affine_map<(d0, d1, d2) -> ()>

// A collapse_shape feeding quantize moves onto the quantized side, leaving the
// quantize next to its real-valued producer.
func.func @absorb_quantize_producer_collapse(%input: tensor<2x3x4xf32>, %scale: f32)
    -> tensor<6x4xi8> {
  %collapsed = tensor.collapse_shape %input [[0, 1], [2]]
      : tensor<2x3x4xf32> into tensor<6x4xf32>
  %init = tensor.empty() : tensor<6x4xi8>
  %quantized = iree_linalg_ext.quantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       quant_min = -128 : i64, quant_max = 127 : i64}
      ins(%collapsed, %scale : tensor<6x4xf32>, f32)
      outs(%init : tensor<6x4xi8>) -> tensor<6x4xi8>
  return %quantized : tensor<6x4xi8>
}
// CHECK-LABEL: func.func @absorb_quantize_producer_collapse(
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<2x3x4xf32>
//       CHECK:   %[[QUANTIZED:.+]] = iree_linalg_ext.quantize_affine
//  CHECK-SAME:     ins(%[[INPUT]]
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[QUANTIZED]]
//       CHECK:   return %[[COLLAPSED]]

// -----

#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#scalar3 = affine_map<(d0, d1, d2) -> ()>

// A reshape consuming a dequantize moves onto the quantized side, so whatever
// consumed the reshape sees the dequantize directly.
func.func @absorb_consumer_expand(%aq: tensor<3x6x6xi8>, %sa: f32, %za: i64) -> tensor<1x3x6x6xf32> {
  %init = tensor.empty() : tensor<3x6x6xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [#id3, #scalar3, #scalar3, #id3]}
      ins(%aq, %sa, %za : tensor<3x6x6xi8>, f32, i64)
      outs(%init : tensor<3x6x6xf32>) -> tensor<3x6x6xf32>
  %expanded = tensor.expand_shape %a [[0, 1], [2], [3]] output_shape [1, 3, 6, 6]
      : tensor<3x6x6xf32> into tensor<1x3x6x6xf32>
  return %expanded : tensor<1x3x6x6xf32>
}
// CHECK-LABEL: func.func @absorb_consumer_expand(
//  CHECK-SAME:     %[[AQ:[a-zA-Z0-9_]+]]: tensor<3x6x6xi8>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[AQ]]
//  CHECK-SAME:     tensor<3x6x6xi8> into tensor<1x3x6x6xi8>
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[EXPANDED]]
//       CHECK:   return %[[DEQ]]

// -----

#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#scalar3 = affine_map<(d0, d1, d2) -> ()>

// With a second reader at the dequantize's own rank the reshape is not absorbed.
// Absorbing it would re-rank the dequantize and leave the second reader behind a
// new reshape, which is a move rather than a removal, and that reader can be the
// contraction this pass exists to keep adjacent. Unit extent folding produces
// exactly this shape.
func.func @decline_consumer_expand_with_second_reader(%aq: tensor<3x6x6xi8>, %sa: f32, %za: i64)
    -> (tensor<1x3x6x6xf32>, tensor<3x6x6xf32>) {
  %init = tensor.empty() : tensor<3x6x6xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [#id3, #scalar3, #scalar3, #id3]}
      ins(%aq, %sa, %za : tensor<3x6x6xi8>, f32, i64)
      outs(%init : tensor<3x6x6xf32>) -> tensor<3x6x6xf32>
  %expanded = tensor.expand_shape %a [[0, 1], [2], [3]] output_shape [1, 3, 6, 6]
      : tensor<3x6x6xf32> into tensor<1x3x6x6xf32>
  return %expanded, %a : tensor<1x3x6x6xf32>, tensor<3x6x6xf32>
}
// CHECK-LABEL: func.func @decline_consumer_expand_with_second_reader(
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     outs(%{{.+}} : tensor<3x6x6xf32>) -> tensor<3x6x6xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[DEQ]]
//       CHECK:   return %[[EXPANDED]], %[[DEQ]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// A transpose consuming a per-channel dequantize moves onto the quantized side,
// which both moves fewer bits and folds into constant weights. The quantization
// parameter map follows the permutation.
func.func @bubble_transpose_through_per_channel_dequantize(%bq: tensor<3x4xi8>, %sb: tensor<3xf32>)
    -> tensor<4x3xf32> {
  %init = tensor.empty() : tensor<3x4xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [#id2, #row2, #id2]}
      ins(%bq, %sb : tensor<3x4xi8>, tensor<3xf32>)
      outs(%init : tensor<3x4xf32>) -> tensor<3x4xf32>
  %tinit = tensor.empty() : tensor<4x3xf32>
  %t = linalg.transpose ins(%b : tensor<3x4xf32>) outs(%tinit : tensor<4x3xf32>)
      permutation = [1, 0]
  return %t : tensor<4x3xf32>
}
// The scale was indexed by the row and is now indexed by the column.
//       CHECK: #[[$SCALE:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL: func.func @bubble_transpose_through_per_channel_dequantize(
//  CHECK-SAME:     %[[BQ:[a-zA-Z0-9_]+]]: tensor<3x4xi8>
//       CHECK:   %[[T:.+]] = linalg.transpose ins(%[[BQ]] : tensor<3x4xi8>)
//  CHECK-SAME:     permutation = [1, 0]
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     #[[$SCALE]]
//  CHECK-SAME:     ins(%[[T]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// Padding a dequantized value with zero is the same as padding the quantized
// value with the zero point, because the zero point is the quantized
// representation of 0.0. Bubbling the pad below the dequantize pads narrower
// data and restores the adjacency.
func.func @bubble_pad_through_asymmetric_dequantize(%aq: tensor<4x4xi8>, %sa: f32, %za: i64)
    -> tensor<6x6xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<4x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [#id2, #scalar2, #scalar2, #id2]}
      ins(%aq, %sa, %za : tensor<4x4xi8>, f32, i64)
      outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %padded = tensor.pad %a low[1, 1] high[1, 1] {
  ^bb0(%i: index, %j: index):
    tensor.yield %cst : f32
  } : tensor<4x4xf32> to tensor<6x6xf32>
  return %padded : tensor<6x6xf32>
}
// CHECK-LABEL: func.func @bubble_pad_through_asymmetric_dequantize(
//  CHECK-SAME:     %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x4xi8>
//  CHECK-SAME:     %[[ZA:[a-zA-Z0-9_]+]]: i64
// The quantized side is padded with the zero point, truncated to the storage type.
//       CHECK:   %[[ZP:.+]] = arith.trunci %[[ZA]] : i64 to i8
//       CHECK:   %[[PADDED:.+]] = tensor.pad %[[AQ]] low[1, 1] high[1, 1]
//       CHECK:     tensor.yield %[[ZP]] : i8
//       CHECK:   } : tensor<4x4xi8> to tensor<6x6xi8>
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[PADDED]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// A symmetric dequantize has an implicit zero point of zero, so the quantized
// side is padded with a zero of the storage type.
func.func @bubble_pad_through_symmetric_dequantize(%aq: tensor<4x4xi8>, %sa: tensor<4xf32>)
    -> tensor<4x6xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<4x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [#id2, #row2, #id2]}
      ins(%aq, %sa : tensor<4x4xi8>, tensor<4xf32>)
      outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %padded = tensor.pad %a low[0, 1] high[0, 1] {
  ^bb0(%i: index, %j: index):
    tensor.yield %cst : f32
  } : tensor<4x4xf32> to tensor<4x6xf32>
  return %padded : tensor<4x6xf32>
}
// CHECK-LABEL: func.func @bubble_pad_through_symmetric_dequantize(
//  CHECK-SAME:     %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x4xi8>
//   CHECK-DAG:   %[[ZERO:.+]] = arith.constant 0 : i8
//       CHECK:   %[[PADDED:.+]] = tensor.pad %[[AQ]] low[0, 1] high[0, 1]
//       CHECK:     tensor.yield %[[ZERO]] : i8
//       CHECK:   } : tensor<4x4xi8> to tensor<4x6xi8>
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[PADDED]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// The padded dimension indexes the scale, so the padded rows have no scale to
// read and the pad cannot move below the dequantize.
func.func @decline_pad_along_quantized_axis(%aq: tensor<4x4xi8>, %sa: tensor<4xf32>)
    -> tensor<6x4xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<4x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [#id2, #row2, #id2]}
      ins(%aq, %sa : tensor<4x4xi8>, tensor<4xf32>)
      outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %padded = tensor.pad %a low[1, 0] high[1, 0] {
  ^bb0(%i: index, %j: index):
    tensor.yield %cst : f32
  } : tensor<4x4xf32> to tensor<6x4xf32>
  return %padded : tensor<6x4xf32>
}
// CHECK-LABEL: func.func @decline_pad_along_quantized_axis(
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//       CHECK:   tensor.pad %[[DEQ]] low[1, 0] high[1, 0]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// A per-channel zero point cannot pad the quantized side, because the whole
// padded region takes a single value while the zero point varies along the
// channel.
func.func @decline_pad_with_per_channel_zero_point(%aq: tensor<4x4xi8>, %sa: tensor<4xf32>,
    %za: tensor<4xi8>) -> tensor<4x6xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<4x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [#id2, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, #id2]}
      ins(%aq, %sa, %za : tensor<4x4xi8>, tensor<4xf32>, tensor<4xi8>)
      outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %padded = tensor.pad %a low[0, 1] high[0, 1] {
  ^bb0(%i: index, %j: index):
    tensor.yield %cst : f32
  } : tensor<4x4xf32> to tensor<4x6xf32>
  return %padded : tensor<4x6xf32>
}
// CHECK-LABEL: func.func @decline_pad_with_per_channel_zero_point(
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//       CHECK:   tensor.pad %[[DEQ]] low[0, 1] high[0, 1]

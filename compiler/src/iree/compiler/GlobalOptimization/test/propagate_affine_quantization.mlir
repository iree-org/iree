// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-global-opt-propagate-affine-quantization))" %s | FileCheck %s

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// A collapse_shape feeding quantize moves onto the quantized side, leaving the
// quantize next to its real-valued producer.
func.func @absorb_quantize_producer_collapse(%input: tensor<2x3x4xf32>, %scale: f32)
    -> tensor<6x4xi8> {
  %collapsed = tensor.collapse_shape %input [[0, 1], [2]]
      : tensor<2x3x4xf32> into tensor<6x4xf32>
  %init = tensor.empty() : tensor<6x4xi8>
  %quantized = iree_linalg_ext.quantize_affine
      <indexing_maps = [#id2, #scalar2, #id2],
       quant_min = -128, quant_max = 127>
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

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>

// Only a collapse_shape on the quantize's value operand is folded. Folding one
// that produces the scale would add an expand_shape on the real-valued input
// rather than remove the collapse_shape.
func.func @decline_quantize_scale_collapse(%input: tensor<6x4xf32>,
    %scale: tensor<2x3xf32>) -> tensor<6x4xi8> {
  %collapsed = tensor.collapse_shape %scale [[0, 1]]
      : tensor<2x3xf32> into tensor<6xf32>
  %init = tensor.empty() : tensor<6x4xi8>
  %quantized = iree_linalg_ext.quantize_affine
      <indexing_maps = [#id2, #row2, #id2],
       quant_min = -128, quant_max = 127>
      ins(%input, %collapsed : tensor<6x4xf32>, tensor<6xf32>)
      outs(%init : tensor<6x4xi8>) -> tensor<6x4xi8>
  return %quantized : tensor<6x4xi8>
}
// CHECK-LABEL: func.func @decline_quantize_scale_collapse(
//  CHECK-SAME:     %[[INPUT:[a-zA-Z0-9_]+]]: tensor<6x4xf32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape
//       CHECK:   iree_linalg_ext.quantize_affine
//  CHECK-SAME:     ins(%[[INPUT]], %[[COLLAPSED]] :

// -----

#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#scalar3 = affine_map<(d0, d1, d2) -> ()>

// An expand_shape consuming a dequantize moves onto the quantized side, so the
// consumers of the expand_shape read the dequantize directly.
func.func @absorb_consumer_expand(%aq: tensor<3x6x6xi8>, %sa: f32, %za: i64) -> tensor<1x3x6x6xf32> {
  %init = tensor.empty() : tensor<3x6x6xf32>
  %a = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id3, #scalar3, #scalar3, #id3]>
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

// An expand_shape is not absorbed when the dequantize has another consumer.
// Expanding the dequantize would put a collapse_shape on that consumer's path
// instead of removing the expand_shape.
func.func @decline_expand_with_multiple_consumers(%aq: tensor<3x6x6xi8>, %sa: f32, %za: i64)
    -> (tensor<1x3x6x6xf32>, tensor<3x6x6xf32>) {
  %init = tensor.empty() : tensor<3x6x6xf32>
  %a = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id3, #scalar3, #scalar3, #id3]>
      ins(%aq, %sa, %za : tensor<3x6x6xi8>, f32, i64)
      outs(%init : tensor<3x6x6xf32>) -> tensor<3x6x6xf32>
  %expanded = tensor.expand_shape %a [[0, 1], [2], [3]] output_shape [1, 3, 6, 6]
      : tensor<3x6x6xf32> into tensor<1x3x6x6xf32>
  return %expanded, %a : tensor<1x3x6x6xf32>, tensor<3x6x6xf32>
}
// CHECK-LABEL: func.func @decline_expand_with_multiple_consumers(
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     outs(%{{.+}} : tensor<3x6x6xf32>) -> tensor<3x6x6xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[DEQ]]
//       CHECK:   return %[[EXPANDED]], %[[DEQ]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// Padding a dequantized value with zero is the same as padding the quantized
// value with the zero point, because the zero point is the quantized
// representation of 0.0. Bubbling the pad above the dequantize pads i8 instead
// of f32 data and makes the dequantize the direct producer of the pad's
// consumer.
func.func @bubble_pad_through_asymmetric_dequantize(%aq: tensor<4x4xi8>, %sa: f32, %za: i64)
    -> tensor<6x6xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<4x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #scalar2, #scalar2, #id2]>
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
#scalar2 = affine_map<(d0, d1) -> ()>

// A pad bubbles through a dequantize that has other users: the pad reads a new
// dequantize of the padded quantized value, and the other users keep the
// original dequantize.
func.func @bubble_pad_through_multi_use_dequantize(%aq: tensor<4x4xi8>, %sa: f32, %za: i64)
    -> (tensor<6x6xf32>, tensor<4x4xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<4x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #scalar2, #scalar2, #id2]>
      ins(%aq, %sa, %za : tensor<4x4xi8>, f32, i64)
      outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %padded = tensor.pad %a low[1, 1] high[1, 1] {
  ^bb0(%i: index, %j: index):
    tensor.yield %cst : f32
  } : tensor<4x4xf32> to tensor<6x6xf32>
  return %padded, %a : tensor<6x6xf32>, tensor<4x4xf32>
}
// CHECK-LABEL: func.func @bubble_pad_through_multi_use_dequantize(
//  CHECK-SAME:     %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x4xi8>
//   CHECK-DAG:   %[[UNPADDED:.+]] = iree_linalg_ext.dequantize_affine <{{.*}}> ins(%[[AQ]],
//   CHECK-DAG:   %[[PADDED_Q:.+]] = tensor.pad %[[AQ]] low[1, 1] high[1, 1]
//   CHECK-DAG:   %[[PADDED:.+]] = iree_linalg_ext.dequantize_affine <{{.*}}> ins(%[[PADDED_Q]],
//       CHECK:   return %[[PADDED]], %[[UNPADDED]]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// A per-tensor zero point may also be a 0-d tensor. Its single element pads
// the quantized side.
func.func @bubble_pad_through_0d_zero_point_dequantize(%aq: tensor<4x4xi8>,
    %sa: tensor<f32>, %za: tensor<i8>) -> tensor<6x6xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<4x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #scalar2, #scalar2, #id2]>
      ins(%aq, %sa, %za : tensor<4x4xi8>, tensor<f32>, tensor<i8>)
      outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %padded = tensor.pad %a low[1, 1] high[1, 1] {
  ^bb0(%i: index, %j: index):
    tensor.yield %cst : f32
  } : tensor<4x4xf32> to tensor<6x6xf32>
  return %padded : tensor<6x6xf32>
}
// CHECK-LABEL: func.func @bubble_pad_through_0d_zero_point_dequantize(
//  CHECK-SAME:     %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x4xi8>
//  CHECK-SAME:     %[[SA:[a-zA-Z0-9_]+]]: tensor<f32>
//  CHECK-SAME:     %[[ZA:[a-zA-Z0-9_]+]]: tensor<i8>
//       CHECK:   %[[ZP:.+]] = tensor.extract %[[ZA]][] : tensor<i8>
//       CHECK:   %[[PADDED:.+]] = tensor.pad %[[AQ]] low[1, 1] high[1, 1]
//       CHECK:     tensor.yield %[[ZP]] : i8
//       CHECK:   } : tensor<4x4xi8> to tensor<6x6xi8>
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[PADDED]], %[[SA]], %[[ZA]] :
//  CHECK-SAME:     -> tensor<6x6xf32>

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// A pad may have a static result type for a dynamic source; the quantized pad
// keeps that static result type.
func.func @bubble_pad_keeps_static_result_type(%aq: tensor<?x4xi8>, %sa: f32, %za: i8)
    -> tensor<6x6xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %aq, %c0 : tensor<?x4xi8>
  %init = tensor.empty(%dim) : tensor<?x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #scalar2, #scalar2, #id2]>
      ins(%aq, %sa, %za : tensor<?x4xi8>, f32, i8)
      outs(%init : tensor<?x4xf32>) -> tensor<?x4xf32>
  %padded = tensor.pad %a low[1, 1] high[1, 1] {
  ^bb0(%i: index, %j: index):
    tensor.yield %cst : f32
  } : tensor<?x4xf32> to tensor<6x6xf32>
  return %padded : tensor<6x6xf32>
}
// CHECK-LABEL: func.func @bubble_pad_keeps_static_result_type(
//  CHECK-SAME:     %[[AQ:[a-zA-Z0-9_]+]]: tensor<?x4xi8>
//       CHECK:   %[[PADDED:.+]] = tensor.pad %[[AQ]] low[1, 1] high[1, 1]
//       CHECK:   } : tensor<?x4xi8> to tensor<6x6xi8>
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[PADDED]]
//  CHECK-SAME:     -> tensor<6x6xf32>

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
      <indexing_maps = [#id2, #row2, #id2]>
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
// read and the pad cannot move above the dequantize.
func.func @decline_pad_along_quantized_axis(%aq: tensor<4x4xi8>, %sa: tensor<4xf32>)
    -> tensor<6x4xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<4x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2]>
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

// A per-channel zero point cannot pad the quantized side: the quantized pad
// would have to read the zero point from its body using the channel index, and
// tensor.pad tiling only supports a padding value defined outside the pad.
func.func @decline_pad_with_per_channel_zero_point(%aq: tensor<4x4xi8>, %sa: tensor<4xf32>,
    %za: tensor<4xi8>) -> tensor<4x6xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<4x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, #id2]>
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

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// Sub-byte storage holds packed constants. A pad of packed values would run at
// runtime with elements that need not start on a byte boundary, so the pad
// stays on the real-valued side.
func.func @decline_pad_sub_byte_storage(%aq: tensor<4x4xi4>, %sa: f32)
    -> tensor<4x6xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<4x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #scalar2, #id2], quant_min = -8, quant_max = 7>
      ins(%aq, %sa : tensor<4x4xi4>, f32)
      outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %padded = tensor.pad %a low[0, 1] high[0, 1] {
  ^bb0(%i: index, %j: index):
    tensor.yield %cst : f32
  } : tensor<4x4xf32> to tensor<4x6xf32>
  return %padded : tensor<4x6xf32>
}
// CHECK-LABEL: func.func @decline_pad_sub_byte_storage(
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//       CHECK:   tensor.pad %[[DEQ]] low[0, 1] high[0, 1]

// -----

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#scalar2 = affine_map<(d0, d1) -> ()>

// Unlike a pad, a reshape keeps the row-major order in which sub-byte values
// are packed, so an expand_shape still moves onto sub-byte storage.
func.func @absorb_consumer_expand_sub_byte_storage(%aq: tensor<4x8xi4>, %sa: f32)
    -> tensor<4x2x4xf32> {
  %init = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #scalar2, #id2], quant_min = -8, quant_max = 7>
      ins(%aq, %sa : tensor<4x8xi4>, f32)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  %expanded = tensor.expand_shape %a [[0], [1, 2]] output_shape [4, 2, 4]
      : tensor<4x8xf32> into tensor<4x2x4xf32>
  return %expanded : tensor<4x2x4xf32>
}
// CHECK-LABEL: func.func @absorb_consumer_expand_sub_byte_storage(
//  CHECK-SAME:     %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x8xi4>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[AQ]]
//  CHECK-SAME:     tensor<4x8xi4> into tensor<4x2x4xi4>
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[EXPANDED]]
//       CHECK:   return %[[DEQ]]

// -----

// Output dimension zero corresponds to quantized input dimension one, so the
// pad on output dimension zero becomes a pad on input dimension one. The
// per-channel scale varies along the other dimension and stays unchanged.
func.func @pad_permuted_output(%q: tensor<2x3xi8>, %scale: tensor<2xf32>,
    %zp: i8) -> tensor<5x2xf32> {
  %init = tensor.empty() : tensor<3x2xf32>
  %d = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d1, d0)>]>
      ins(%q, %scale, %zp : tensor<2x3xi8>, tensor<2xf32>, i8)
      outs(%init : tensor<3x2xf32>) -> tensor<3x2xf32>
  %zero = arith.constant 0.0 : f32
  %p = tensor.pad %d low[1, 0] high[1, 0] {
  ^bb0(%i: index, %j: index):
    tensor.yield %zero : f32
  } : tensor<3x2xf32> to tensor<5x2xf32>
  return %p : tensor<5x2xf32>
}
// CHECK-LABEL: func.func @pad_permuted_output(
//  CHECK-SAME:     %[[Q:[a-zA-Z0-9_]+]]: tensor<2x3xi8>
//       CHECK:   %[[PADDED:.+]] = tensor.pad %[[Q]] low[0, 1] high[0, 1]
//       CHECK:     tensor<2x3xi8> to tensor<2x5xi8>
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[PADDED]]
//  CHECK-SAME:     outs(%{{.+}} : tensor<5x2xf32>) -> tensor<5x2xf32>

// -----

// The control function admits a generic transpose of a dequantized value, so it
// is specialized and then fused into the dequantize in the same rewrite.
func.func @generic_transpose_after_dequantize(%q: tensor<2x3xi8>, %scale: f32)
    -> tensor<3x2xf32> {
  %init = tensor.empty() : tensor<2x3xf32>
  %d = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]>
      ins(%q, %scale : tensor<2x3xi8>, f32)
      outs(%init : tensor<2x3xf32>) -> tensor<2x3xf32>
  %transpose_init = tensor.empty() : tensor<3x2xf32>
  %t = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%d : tensor<2x3xf32>) outs(%transpose_init : tensor<3x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x2xf32>
  return %t : tensor<3x2xf32>
}
// CHECK-LABEL: func.func @generic_transpose_after_dequantize(
//   CHECK-NOT:   linalg.generic
//   CHECK-NOT:   linalg.transpose
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     -> tensor<3x2xf32>
//   CHECK-NOT:   linalg.generic
//   CHECK-NOT:   linalg.transpose
//       CHECK:   return %[[DEQ]]

// -----

#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#block3 = affine_map<(d0, d1, d2) -> (d0, d1)>

// Blockwise operands as produced by the Torch frontend for a linear layer:
// both dequantize over (rows, blocks, block elements) and collapse the blocks
// back into K, and the weight is transposed. The transpose absorbs the weight
// collapse_shape and folds into the dequantize, then the matmul absorbs both
// collapse_shapes and iterates over (m, n, g, l), reading the dequantizes
// directly.
func.func @expand_matmul_over_collapsed_dequantizes(
    %aq: tensor<?x2x32xi8>, %sa: tensor<?x2xf32>, %za: tensor<?x2xi64>,
    %bq: tensor<8x2x32xi8>, %sb: tensor<8x2xf32>, %zb: tensor<8x2xi8>)
    -> tensor<?x8xf32> {
  %c0 = arith.constant 0 : index
  %m = tensor.dim %aq, %c0 : tensor<?x2x32xi8>
  %ainit = tensor.empty(%m) : tensor<?x2x32xf32>
  %a = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id3, #block3, #block3, #id3],
       quant_min = -128, quant_max = 127>
      ins(%aq, %sa, %za : tensor<?x2x32xi8>, tensor<?x2xf32>, tensor<?x2xi64>)
      outs(%ainit : tensor<?x2x32xf32>) -> tensor<?x2x32xf32>
  %ac = tensor.collapse_shape %a [[0], [1, 2]]
      : tensor<?x2x32xf32> into tensor<?x64xf32>
  %binit = tensor.empty() : tensor<8x2x32xf32>
  %b = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id3, #block3, #block3, #id3],
       quant_min = 0, quant_max = 255, input_unsigned, zp_unsigned>
      ins(%bq, %sb, %zb : tensor<8x2x32xi8>, tensor<8x2xf32>, tensor<8x2xi8>)
      outs(%binit : tensor<8x2x32xf32>) -> tensor<8x2x32xf32>
  %bc = tensor.collapse_shape %b [[0], [1, 2]]
      : tensor<8x2x32xf32> into tensor<8x64xf32>
  %tinit = tensor.empty() : tensor<64x8xf32>
  %bt = linalg.transpose ins(%bc : tensor<8x64xf32>)
      outs(%tinit : tensor<64x8xf32>) permutation = [1, 0]
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty(%m) : tensor<?x8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x8xf32>)
      -> tensor<?x8xf32>
  %c = linalg.matmul ins(%ac, %bt : tensor<?x64xf32>, tensor<64x8xf32>)
      outs(%fill : tensor<?x8xf32>) -> tensor<?x8xf32>
  return %c : tensor<?x8xf32>
}
//   CHECK-DAG: #[[$LHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//   CHECK-DAG: #[[$RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
//   CHECK-DAG: #[[$OUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
//   CHECK-DAG: #[[$TRANSPOSED:.+]] = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
// CHECK-LABEL: func.func @expand_matmul_over_collapsed_dequantizes(
//   CHECK-NOT:   tensor.collapse_shape
//   CHECK-NOT:   linalg.transpose
//       CHECK:   %[[A:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     -> tensor<?x2x32xf32>
//       CHECK:   %[[B:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     #[[$TRANSPOSED]]]
//  CHECK-SAME:     -> tensor<2x32x8xf32>
//       CHECK:   %[[C:.+]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[$LHS]], #[[$RHS]], #[[$OUT]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction", "reduction"]
//  CHECK-SAME:     ins(%[[A]], %[[B]] : tensor<?x2x32xf32>, tensor<2x32x8xf32>)
//   CHECK-NOT:   tensor.collapse_shape
//       CHECK:   return %[[C]]

// -----

#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#scalar3 = affine_map<(d0, d1, d2) -> ()>

// A transpose of a collapsed dequantize moves the collapse_shape after itself,
// so the transpose folds into the dequantize's output map.
func.func @transpose_absorbs_collapsed_dequantize(%q: tensor<8x2x32xi8>,
    %scale: f32) -> tensor<64x8xf32> {
  %init = tensor.empty() : tensor<8x2x32xf32>
  %d = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id3, #scalar3, #id3]>
      ins(%q, %scale : tensor<8x2x32xi8>, f32)
      outs(%init : tensor<8x2x32xf32>) -> tensor<8x2x32xf32>
  %collapsed = tensor.collapse_shape %d [[0], [1, 2]]
      : tensor<8x2x32xf32> into tensor<8x64xf32>
  %tinit = tensor.empty() : tensor<64x8xf32>
  %t = linalg.transpose ins(%collapsed : tensor<8x64xf32>)
      outs(%tinit : tensor<64x8xf32>) permutation = [1, 0]
  return %t : tensor<64x8xf32>
}
// CHECK-LABEL: func.func @transpose_absorbs_collapsed_dequantize(
//   CHECK-NOT:   linalg.transpose
//       CHECK:   %[[DEQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     -> tensor<2x32x8xf32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[DEQ]] {{\[}}[0, 1], [2]]
//  CHECK-SAME:     tensor<2x32x8xf32> into tensor<64x8xf32>
//       CHECK:   return %[[COLLAPSED]]

// -----

#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#scalar3 = affine_map<(d0, d1, d2) -> ()>

// Expanding the matmul would reshape the real-valued rhs without producing a
// contraction of dequantized operands, so the collapse_shape stays.
func.func @decline_expand_matmul_with_real_operand(%aq: tensor<4x2x32xi8>,
    %sa: f32, %b: tensor<64x8xf32>) -> tensor<4x8xf32> {
  %ainit = tensor.empty() : tensor<4x2x32xf32>
  %a = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id3, #scalar3, #id3]>
      ins(%aq, %sa : tensor<4x2x32xi8>, f32)
      outs(%ainit : tensor<4x2x32xf32>) -> tensor<4x2x32xf32>
  %ac = tensor.collapse_shape %a [[0], [1, 2]]
      : tensor<4x2x32xf32> into tensor<4x64xf32>
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<4x8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<4x8xf32>)
      -> tensor<4x8xf32>
  %c = linalg.matmul ins(%ac, %b : tensor<4x64xf32>, tensor<64x8xf32>)
      outs(%fill : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %c : tensor<4x8xf32>
}
// CHECK-LABEL: func.func @decline_expand_matmul_with_real_operand(
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape
//       CHECK:   linalg.matmul ins(%[[COLLAPSED]],

// -----

#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#scalar3 = affine_map<(d0, d1, d2) -> ()>

// Only contractions and transposes absorb the collapse_shape; an elementwise
// consumer is left to the general reshape propagation.
func.func @decline_expand_elementwise_consumer(%q: tensor<4x2x32xi8>,
    %scale: f32) -> tensor<4x64xf32> {
  %init = tensor.empty() : tensor<4x2x32xf32>
  %d = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id3, #scalar3, #id3]>
      ins(%q, %scale : tensor<4x2x32xi8>, f32)
      outs(%init : tensor<4x2x32xf32>) -> tensor<4x2x32xf32>
  %collapsed = tensor.collapse_shape %d [[0], [1, 2]]
      : tensor<4x2x32xf32> into tensor<4x64xf32>
  %einit = tensor.empty() : tensor<4x64xf32>
  %e = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%collapsed : tensor<4x64xf32>) outs(%einit : tensor<4x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %neg = arith.negf %in : f32
      linalg.yield %neg : f32
    } -> tensor<4x64xf32>
  return %e : tensor<4x64xf32>
}
// CHECK-LABEL: func.func @decline_expand_elementwise_consumer(
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[COLLAPSED]] :

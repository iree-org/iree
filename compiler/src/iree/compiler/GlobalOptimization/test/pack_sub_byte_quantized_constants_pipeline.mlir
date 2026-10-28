// RUN: iree-opt --split-input-file --iree-global-optimization-transformation-pipeline --iree-global-opt-enable-sub-byte-quantized-storage=true %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-global-optimization-transformation-pipeline --iree-global-opt-enable-sub-byte-quantized-storage=false %s | FileCheck %s --check-prefix=DISABLED
// RUN: iree-opt --split-input-file --iree-global-optimization-transformation-pipeline="const-expr-hoisting=false" --iree-global-opt-enable-sub-byte-quantized-storage=true %s | FileCheck %s --check-prefix=DISABLED

// Exercises the pass in the production pipeline: the packing is hoisted into
// a half-size i8 global, and the integer contraction reads the bitcast i4
// value. Without hoisting the packing would run on every call, so the pass does
// not run.

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#row2 = affine_map<(d0, d1) -> (d0)>
#scalar2 = affine_map<(d0, d1) -> ()>

// PT2E linear layer: a float weight quantized to [-8, 7] and transposed for
// the matmul, with per-tensor quantized activations.
util.global private @weight = dense<0.5> : tensor<8x16xf32>
util.func public @per_channel_signed_linear(%x: tensor<4x16xf32>) -> tensor<4x8xf32> {
  %weight = util.global.load @weight : tensor<8x16xf32>
  %weight_scale = arith.constant dense<0.125> : tensor<8xf32>
  %x_scale = arith.constant 0.25 : f32
  %xq_init = tensor.empty() : tensor<4x16xi8>
  %xq = iree_linalg_ext.quantize_affine
      <indexing_maps = [#id2, #scalar2, #id2], quant_min = -128, quant_max = 127>
      ins(%x, %x_scale : tensor<4x16xf32>, f32)
      outs(%xq_init : tensor<4x16xi8>) -> tensor<4x16xi8>
  %xdq_init = tensor.empty() : tensor<4x16xf32>
  %xdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #scalar2, #id2], quant_min = -128, quant_max = 127>
      ins(%xq, %x_scale : tensor<4x16xi8>, f32)
      outs(%xdq_init : tensor<4x16xf32>) -> tensor<4x16xf32>
  %wq_init = tensor.empty() : tensor<8x16xi8>
  %wq = iree_linalg_ext.quantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%weight, %weight_scale : tensor<8x16xf32>, tensor<8xf32>)
      outs(%wq_init : tensor<8x16xi8>) -> tensor<8x16xi8>
  %wdq_init = tensor.empty() : tensor<8x16xf32>
  %wdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [#id2, #row2, #id2], quant_min = -8, quant_max = 7>
      ins(%wq, %weight_scale : tensor<8x16xi8>, tensor<8xf32>)
      outs(%wdq_init : tensor<8x16xf32>) -> tensor<8x16xf32>
  %wt_init = tensor.empty() : tensor<16x8xf32>
  %wt = linalg.transpose ins(%wdq : tensor<8x16xf32>)
      outs(%wt_init : tensor<16x8xf32>) permutation = [1, 0]
  %zero = arith.constant 0.0 : f32
  %acc_init = tensor.empty() : tensor<4x8xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%acc_init : tensor<4x8xf32>) -> tensor<4x8xf32>
  %result = linalg.matmul ins(%xdq, %wt : tensor<4x16xf32>, tensor<16x8xf32>)
      outs(%acc : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %result : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @per_channel_signed_linear(
//       CHECK:   %[[PACKED:.+]] = util.global.load immutable @{{.+}} : tensor<8x8xi8>
//       CHECK:   %[[WEIGHT:.+]] = flow.tensor.bitcast %[[PACKED]] : tensor<8x8xi8> -> tensor<8x16xi4>
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%{{.+}}, %[[WEIGHT]] : tensor<4x16xi8>, tensor<8x16xi4>)
//       CHECK:     arith.extsi %{{.+}} : i4 to i32
// DISABLED-LABEL: util.func public @per_channel_signed_linear(
//   DISABLED-NOT:   flow.tensor.bitcast
//       DISABLED:   linalg.generic
//  DISABLED-SAME:     ins(%{{.+}}, %{{.+}} : tensor<4x16xi8>, tensor<8x16xi8>)

// -----

// MatMulNBits form: unsigned 4-bit weights quantized per block with the
// default zero point 8, collapsed back into K and transposed. Propagation first
// expands the matmul over the blocks, so the weight keeps its [N, K/B, B]
// shape and is packed along B. Packing shifts the weights to signed i4, which
// drops the zero point, so the contraction sign-extends the weights and needs
// no correction for a weight zero point.
util.global private @weight = dense<9> : tensor<8x2x32xi8>
util.func public @blockwise_unsigned_linear(
    %aq: tensor<4x2x32xi8>, %sa: tensor<4x2xf32>, %sb: tensor<8x2xf32>)
    -> tensor<4x8xf32> {
  %ai = tensor.empty() : tensor<4x2x32xf32>
  %a = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(m, g, l) -> (m, g, l)>,
                        affine_map<(m, g, l) -> (m, g)>,
                        affine_map<(m, g, l) -> (m, g, l)>],
       quant_min = -128, quant_max = 127>
      ins(%aq, %sa : tensor<4x2x32xi8>, tensor<4x2xf32>)
      outs(%ai : tensor<4x2x32xf32>) -> tensor<4x2x32xf32>
  %ac = tensor.collapse_shape %a [[0], [1, 2]]
      : tensor<4x2x32xf32> into tensor<4x64xf32>
  %bq = util.global.load @weight : tensor<8x2x32xi8>
  %zb = arith.constant dense<8> : tensor<8x2xi8>
  %bi = tensor.empty() : tensor<8x2x32xf32>
  %b = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(n, g, l) -> (n, g, l)>,
                        affine_map<(n, g, l) -> (n, g)>,
                        affine_map<(n, g, l) -> (n, g)>,
                        affine_map<(n, g, l) -> (n, g, l)>],
       quant_min = 0, quant_max = 15, input_unsigned, zp_unsigned>
      ins(%bq, %sb, %zb : tensor<8x2x32xi8>, tensor<8x2xf32>, tensor<8x2xi8>)
      outs(%bi : tensor<8x2x32xf32>) -> tensor<8x2x32xf32>
  %bc = tensor.collapse_shape %b [[0], [1, 2]]
      : tensor<8x2x32xf32> into tensor<8x64xf32>
  %ti = tensor.empty() : tensor<64x8xf32>
  %bt = linalg.transpose ins(%bc : tensor<8x64xf32>)
      outs(%ti : tensor<64x8xf32>) permutation = [1, 0]
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %result = linalg.matmul ins(%ac, %bt : tensor<4x64xf32>, tensor<64x8xf32>)
      outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  util.return %result : tensor<4x8xf32>
}
// CHECK-LABEL: util.func public @blockwise_unsigned_linear(
//       CHECK:   %[[PACKED:.+]] = util.global.load immutable @{{.+}} : tensor<8x2x16xi8>
//       CHECK:   %[[WEIGHT:.+]] = flow.tensor.bitcast %[[PACKED]] : tensor<8x2x16xi8> -> tensor<8x2x32xi4>
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%{{.+}}, %[[WEIGHT]] : tensor<4x2x32xi8>, tensor<8x2x32xi4>)
//       CHECK:     arith.extsi %{{.+}} : i4 to i32
// The activation row sums a weight zero point would need.
//   CHECK-NOT:   tensor<4x2xi32>
//       CHECK:   util.return
// DISABLED-LABEL: util.func public @blockwise_unsigned_linear(
//   DISABLED-NOT:   flow.tensor.bitcast
//       DISABLED:   linalg.generic
//  DISABLED-SAME:     ins(%{{.+}}, %{{.+}} : tensor<4x2x32xi8>, tensor<8x2x32xi8>)

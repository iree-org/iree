// RUN: iree-opt --split-input-file --mlir-print-local-scope --iree-global-optimization-transformation-pipeline --iree-global-opt-enable-qdq-to-integer-math=false %s | FileCheck %s --check-prefix=STAGE3 --implicit-check-not=linalg.broadcast
// RUN: iree-opt --split-input-file --mlir-print-local-scope --iree-global-optimization-transformation-pipeline --iree-global-opt-enable-qdq-to-integer-math=true %s | FileCheck %s --check-prefix=STAGE4 --implicit-check-not=linalg.broadcast --implicit-check-not=iree_linalg_ext.dequantize_affine

// Stage-boundary tests for contractions over shared quantized weights, run
// through the whole global optimization pipeline.
//
// STAGE3 (QDQ conversion disabled) checks the operand normalization
// post-condition: the contraction reads a dequantize of the weights at their
// native rank through its indexing map, with no broadcast or transpose op on
// the weight's path and no batch-sized dequantize.
//
// STAGE4 (QDQ conversion enabled) checks the integer rewrite post-condition:
// the integer contraction, and any zero-point correction, reads the native
// quantized weights directly.
//
// Weights broadcast after dequantization (form A) and before it (form B)
// reach the same form.

// STAGE3-LABEL: @form_a(
//  STAGE3-SAME:   %[[W:[a-zA-Z0-9]+]]: tensor<16x12xi8>
//       STAGE3:   %[[W_B:.+]] = iree_tensor_ext.compute_barrier.start %[[W]]
//       STAGE3:   %[[A_DQ:.+]] = iree_linalg_ext.dequantize_affine
//       STAGE3:   %[[W_DQ:.+]] = iree_linalg_ext.dequantize_affine
//  STAGE3-SAME:     <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>]>
//  STAGE3-SAME:     ins(%[[W_B]], %{{.+}} : tensor<16x12xi8>, tensor<12xf32>)
//  STAGE3-SAME:     -> tensor<16x12xf32>
//   STAGE3-NOT:   iree_linalg_ext.dequantize_affine
//       STAGE3:   linalg.batch_matmul
//  STAGE3-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  STAGE3-SAME:     ins(%[[A_DQ]], %[[W_DQ]] : tensor<2x8x16xf32>, tensor<16x12xf32>)

// STAGE4-LABEL: @form_a(
//  STAGE4-SAME:   %[[W:[a-zA-Z0-9]+]]: tensor<16x12xi8>
//       STAGE4:   %[[W_B:.+]] = iree_tensor_ext.compute_barrier.start %[[W]]
//       STAGE4:   %[[A_B:.+]] = iree_tensor_ext.compute_barrier.start
//       STAGE4:   linalg.generic
//  STAGE4-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  STAGE4-SAME:     ins(%[[A_B]], %[[W_B]] : tensor<2x8x16xi8>, tensor<16x12xi8>)

// Form A: the weights are dequantized at their native rank, then broadcast.
util.func public @form_a(%a: tensor<2x8x16xi8>, %w: tensor<16x12xi8>, %sa: f32, %sw: tensor<12xf32>) -> tensor<2x8x12xf32> {
  %ae = tensor.empty() : tensor<2x8x16xf32>
  %adq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%a, %sa : tensor<2x8x16xi8>, f32) outs(%ae : tensor<2x8x16xf32>) -> tensor<2x8x16xf32>
  %wfe = tensor.empty() : tensor<16x12xf32>
  %wdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>]>
      ins(%w, %sw : tensor<16x12xi8>, tensor<12xf32>) outs(%wfe : tensor<16x12xf32>) -> tensor<16x12xf32>
  %we = tensor.empty() : tensor<2x16x12xf32>
  %wb = linalg.broadcast ins(%wdq : tensor<16x12xf32>) outs(%we : tensor<2x16x12xf32>) dimensions = [0]
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty() : tensor<2x8x12xf32>
  %init = linalg.fill ins(%zero : f32) outs(%ce : tensor<2x8x12xf32>) -> tensor<2x8x12xf32>
  %result = linalg.batch_matmul ins(%adq, %wb : tensor<2x8x16xf32>, tensor<2x16x12xf32>) outs(%init : tensor<2x8x12xf32>) -> tensor<2x8x12xf32>
  util.return %result : tensor<2x8x12xf32>
}

// -----

// STAGE3-LABEL: @form_b(
//  STAGE3-SAME:   %[[W:[a-zA-Z0-9]+]]: tensor<16x12xi8>
//       STAGE3:   %[[W_B:.+]] = iree_tensor_ext.compute_barrier.start %[[W]]
//       STAGE3:   %[[A_DQ:.+]] = iree_linalg_ext.dequantize_affine
//       STAGE3:   %[[W_DQ:.+]] = iree_linalg_ext.dequantize_affine
//  STAGE3-SAME:     <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>]>
//  STAGE3-SAME:     ins(%[[W_B]], %{{.+}} : tensor<16x12xi8>, tensor<12xf32>)
//  STAGE3-SAME:     -> tensor<16x12xf32>
//   STAGE3-NOT:   iree_linalg_ext.dequantize_affine
//       STAGE3:   linalg.batch_matmul
//  STAGE3-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  STAGE3-SAME:     ins(%[[A_DQ]], %[[W_DQ]] : tensor<2x8x16xf32>, tensor<16x12xf32>)

// STAGE4-LABEL: @form_b(
//  STAGE4-SAME:   %[[W:[a-zA-Z0-9]+]]: tensor<16x12xi8>
//       STAGE4:   %[[W_B:.+]] = iree_tensor_ext.compute_barrier.start %[[W]]
//       STAGE4:   %[[A_B:.+]] = iree_tensor_ext.compute_barrier.start
//       STAGE4:   linalg.generic
//  STAGE4-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  STAGE4-SAME:     ins(%[[A_B]], %[[W_B]] : tensor<2x8x16xi8>, tensor<16x12xi8>)

// Form B: the integer weights are broadcast, then dequantized.
util.func public @form_b(%a: tensor<2x8x16xi8>, %w: tensor<16x12xi8>, %sa: f32, %sw: tensor<12xf32>) -> tensor<2x8x12xf32> {
  %ae = tensor.empty() : tensor<2x8x16xf32>
  %adq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%a, %sa : tensor<2x8x16xi8>, f32) outs(%ae : tensor<2x8x16xf32>) -> tensor<2x8x16xf32>
  %we = tensor.empty() : tensor<2x16x12xi8>
  %wb = linalg.broadcast ins(%w : tensor<16x12xi8>) outs(%we : tensor<2x16x12xi8>) dimensions = [0]
  %wfe = tensor.empty() : tensor<2x16x12xf32>
  %wdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%wb, %sw : tensor<2x16x12xi8>, tensor<12xf32>) outs(%wfe : tensor<2x16x12xf32>) -> tensor<2x16x12xf32>
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty() : tensor<2x8x12xf32>
  %init = linalg.fill ins(%zero : f32) outs(%ce : tensor<2x8x12xf32>) -> tensor<2x8x12xf32>
  %result = linalg.batch_matmul ins(%adq, %wdq : tensor<2x8x16xf32>, tensor<2x16x12xf32>) outs(%init : tensor<2x8x12xf32>) -> tensor<2x8x12xf32>
  util.return %result : tensor<2x8x12xf32>
}

// -----

// The weight's column sums for the activation zero-point correction are
// computed once over the native weights, not per batch element.
// STAGE3-LABEL: @form_b_asymmetric(
//  STAGE3-SAME:   %[[W:[a-zA-Z0-9]+]]: tensor<16x12xi8>
//       STAGE3:   %[[W_B:.+]] = iree_tensor_ext.compute_barrier.start %[[W]]
//       STAGE3:   %[[A_DQ:.+]] = iree_linalg_ext.dequantize_affine
//       STAGE3:   %[[W_DQ:.+]] = iree_linalg_ext.dequantize_affine
//  STAGE3-SAME:     <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>]>
//  STAGE3-SAME:     ins(%[[W_B]], %{{.+}}, %{{.+}} : tensor<16x12xi8>, tensor<12xf32>, tensor<12xi8>)
//   STAGE3-NOT:   iree_linalg_ext.dequantize_affine
//       STAGE3:   linalg.batch_matmul
//  STAGE3-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  STAGE3-SAME:     ins(%[[A_DQ]], %[[W_DQ]] : tensor<2x8x16xf32>, tensor<16x12xf32>)

// STAGE4-LABEL: @form_b_asymmetric(
//  STAGE4-SAME:   %[[W:[a-zA-Z0-9]+]]: tensor<16x12xi8>
//       STAGE4:   %[[W_B:.+]] = iree_tensor_ext.compute_barrier.start %[[W]]
//       STAGE4:   %[[A_B:.+]] = iree_tensor_ext.compute_barrier.start
//       STAGE4:   linalg.generic
//  STAGE4-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  STAGE4-SAME:     ins(%[[A_B]], %[[W_B]] : tensor<2x8x16xi8>, tensor<16x12xi8>)
//       STAGE4:   linalg.generic
//  STAGE4-SAME:     ins(%[[A_B]] : tensor<2x8x16xi8>) outs(%{{.+}} : tensor<2x8xi32>)
//       STAGE4:   linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>]
//  STAGE4-SAME:     ins(%[[W_B]] : tensor<16x12xi8>) outs(%{{.+}} : tensor<12xi32>)

// Form B with a per-channel weight zero point.
util.func public @form_b_asymmetric(%a: tensor<2x8x16xi8>, %w: tensor<16x12xi8>, %sa: f32, %za: i8, %sw: tensor<12xf32>, %zw: tensor<12xi8>) -> tensor<2x8x12xf32> {
  %ae = tensor.empty() : tensor<2x8x16xf32>
  %adq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%a, %sa, %za : tensor<2x8x16xi8>, f32, i8) outs(%ae : tensor<2x8x16xf32>) -> tensor<2x8x16xf32>
  %we = tensor.empty() : tensor<2x16x12xi8>
  %wb = linalg.broadcast ins(%w : tensor<16x12xi8>) outs(%we : tensor<2x16x12xi8>) dimensions = [0]
  %wfe = tensor.empty() : tensor<2x16x12xf32>
  %wdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%wb, %sw, %zw : tensor<2x16x12xi8>, tensor<12xf32>, tensor<12xi8>) outs(%wfe : tensor<2x16x12xf32>) -> tensor<2x16x12xf32>
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty() : tensor<2x8x12xf32>
  %init = linalg.fill ins(%zero : f32) outs(%ce : tensor<2x8x12xf32>) -> tensor<2x8x12xf32>
  %result = linalg.batch_matmul ins(%adq, %wdq : tensor<2x8x16xf32>, tensor<2x16x12xf32>) outs(%init : tensor<2x8x12xf32>) -> tensor<2x8x12xf32>
  util.return %result : tensor<2x8x12xf32>
}

// -----

// The transpose folds into the dequantize's output map, so the named op reads
// [K, N] weights.
// STAGE3-LABEL: @form_a_transposed(
//  STAGE3-SAME:   %[[W:[a-zA-Z0-9]+]]: tensor<12x16xi8>
//       STAGE3:   %[[W_B:.+]] = iree_tensor_ext.compute_barrier.start %[[W]]
//       STAGE3:   %[[A_DQ:.+]] = iree_linalg_ext.dequantize_affine
//       STAGE3:   %[[W_DQ:.+]] = iree_linalg_ext.dequantize_affine
//  STAGE3-SAME:     <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1, d0)>]>
//  STAGE3-SAME:     ins(%[[W_B]], %{{.+}} : tensor<12x16xi8>, tensor<12xf32>)
//  STAGE3-SAME:     -> tensor<16x12xf32>
//   STAGE3-NOT:   iree_linalg_ext.dequantize_affine
//       STAGE3:   linalg.batch_matmul
//  STAGE3-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  STAGE3-SAME:     ins(%[[A_DQ]], %[[W_DQ]] : tensor<2x8x16xf32>, tensor<16x12xf32>)

// STAGE4-LABEL: @form_a_transposed(
//  STAGE4-SAME:   %[[W:[a-zA-Z0-9]+]]: tensor<12x16xi8>
//       STAGE4:   %[[W_B:.+]] = iree_tensor_ext.compute_barrier.start %[[W]]
//       STAGE4:   %[[A_B:.+]] = iree_tensor_ext.compute_barrier.start
//       STAGE4:   linalg.generic
//  STAGE4-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  STAGE4-SAME:     ins(%[[A_B]], %[[W_B]] : tensor<2x8x16xi8>, tensor<12x16xi8>)

// Form A with [N, K] weights: dequantize, transpose, then broadcast.
util.func public @form_a_transposed(%a: tensor<2x8x16xi8>, %w: tensor<12x16xi8>, %sa: f32, %sw: tensor<12xf32>) -> tensor<2x8x12xf32> {
  %ae = tensor.empty() : tensor<2x8x16xf32>
  %adq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%a, %sa : tensor<2x8x16xi8>, f32) outs(%ae : tensor<2x8x16xf32>) -> tensor<2x8x16xf32>
  %wfe = tensor.empty() : tensor<12x16xf32>
  %wdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>]>
      ins(%w, %sw : tensor<12x16xi8>, tensor<12xf32>) outs(%wfe : tensor<12x16xf32>) -> tensor<12x16xf32>
  %te = tensor.empty() : tensor<16x12xf32>
  %wt = linalg.transpose ins(%wdq : tensor<12x16xf32>) outs(%te : tensor<16x12xf32>) permutation = [1, 0]
  %we = tensor.empty() : tensor<2x16x12xf32>
  %wb = linalg.broadcast ins(%wt : tensor<16x12xf32>) outs(%we : tensor<2x16x12xf32>) dimensions = [0]
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty() : tensor<2x8x12xf32>
  %init = linalg.fill ins(%zero : f32) outs(%ce : tensor<2x8x12xf32>) -> tensor<2x8x12xf32>
  %result = linalg.batch_matmul ins(%adq, %wb : tensor<2x8x16xf32>, tensor<2x16x12xf32>) outs(%init : tensor<2x8x12xf32>) -> tensor<2x8x12xf32>
  util.return %result : tensor<2x8x12xf32>
}

// -----

// The transpose and broadcast merge into one producer that folds into the
// contraction map. The named op rejects a batch-broadcast [N, K] RHS, so the
// contraction is a generic over the native [N, K] weights.
// STAGE3-LABEL: @form_b_transposed(
//  STAGE3-SAME:   %[[W:[a-zA-Z0-9]+]]: tensor<12x16xi8>
//       STAGE3:   %[[W_B:.+]] = iree_tensor_ext.compute_barrier.start %[[W]]
//       STAGE3:   %[[A_DQ:.+]] = iree_linalg_ext.dequantize_affine
//       STAGE3:   %[[W_DQ:.+]] = iree_linalg_ext.dequantize_affine
//  STAGE3-SAME:     <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>]>
//  STAGE3-SAME:     ins(%[[W_B]], %{{.+}} : tensor<12x16xi8>, tensor<12xf32>)
//  STAGE3-SAME:     -> tensor<12x16xf32>
//   STAGE3-NOT:   iree_linalg_ext.dequantize_affine
//       STAGE3:   linalg.generic
//  STAGE3-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  STAGE3-SAME:     ins(%[[A_DQ]], %[[W_DQ]] : tensor<2x8x16xf32>, tensor<12x16xf32>)

// STAGE4-LABEL: @form_b_transposed(
//  STAGE4-SAME:   %[[W:[a-zA-Z0-9]+]]: tensor<12x16xi8>
//       STAGE4:   %[[W_B:.+]] = iree_tensor_ext.compute_barrier.start %[[W]]
//       STAGE4:   %[[A_B:.+]] = iree_tensor_ext.compute_barrier.start
//       STAGE4:   linalg.generic
//  STAGE4-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  STAGE4-SAME:     ins(%[[A_B]], %[[W_B]] : tensor<2x8x16xi8>, tensor<12x16xi8>)

// Form B with [N, K] weights: transpose, broadcast, then dequantize.
util.func public @form_b_transposed(%a: tensor<2x8x16xi8>, %w: tensor<12x16xi8>, %sa: f32, %sw: tensor<12xf32>) -> tensor<2x8x12xf32> {
  %ae = tensor.empty() : tensor<2x8x16xf32>
  %adq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%a, %sa : tensor<2x8x16xi8>, f32) outs(%ae : tensor<2x8x16xf32>) -> tensor<2x8x16xf32>
  %te = tensor.empty() : tensor<16x12xi8>
  %wt = linalg.transpose ins(%w : tensor<12x16xi8>) outs(%te : tensor<16x12xi8>) permutation = [1, 0]
  %we = tensor.empty() : tensor<2x16x12xi8>
  %wb = linalg.broadcast ins(%wt : tensor<16x12xi8>) outs(%we : tensor<2x16x12xi8>) dimensions = [0]
  %wfe = tensor.empty() : tensor<2x16x12xf32>
  %wdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%wb, %sw : tensor<2x16x12xi8>, tensor<12xf32>) outs(%wfe : tensor<2x16x12xf32>) -> tensor<2x16x12xf32>
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty() : tensor<2x8x12xf32>
  %init = linalg.fill ins(%zero : f32) outs(%ce : tensor<2x8x12xf32>) -> tensor<2x8x12xf32>
  %result = linalg.batch_matmul ins(%adq, %wdq : tensor<2x8x16xf32>, tensor<2x16x12xf32>) outs(%init : tensor<2x8x12xf32>) -> tensor<2x8x12xf32>
  util.return %result : tensor<2x8x12xf32>
}

// -----

// STAGE3-LABEL: @form_b_dynamic_batch(
//  STAGE3-SAME:   %[[W:[a-zA-Z0-9]+]]: tensor<16x12xi8>
//       STAGE3:   %[[W_B:.+]] = iree_tensor_ext.compute_barrier.start %[[W]]
//       STAGE3:   %[[A_DQ:.+]] = iree_linalg_ext.dequantize_affine
//       STAGE3:   %[[W_DQ:.+]] = iree_linalg_ext.dequantize_affine
//  STAGE3-SAME:     ins(%[[W_B]], %{{.+}} : tensor<16x12xi8>, tensor<12xf32>)
//  STAGE3-SAME:     -> tensor<16x12xf32>
//   STAGE3-NOT:   iree_linalg_ext.dequantize_affine
//       STAGE3:   linalg.batch_matmul
//  STAGE3-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  STAGE3-SAME:     ins(%[[A_DQ]], %[[W_DQ]] : tensor<?x8x16xf32>, tensor<16x12xf32>)

// STAGE4-LABEL: @form_b_dynamic_batch(
//  STAGE4-SAME:   %[[W:[a-zA-Z0-9]+]]: tensor<16x12xi8>
//       STAGE4:   %[[W_B:.+]] = iree_tensor_ext.compute_barrier.start %[[W]]
//       STAGE4:   %[[A_B:.+]] = iree_tensor_ext.compute_barrier.start
//       STAGE4:   linalg.generic
//  STAGE4-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  STAGE4-SAME:     ins(%[[A_B]], %[[W_B]] : tensor<?x8x16xi8>, tensor<16x12xi8>)

// Form B with a dynamic batch.
util.func public @form_b_dynamic_batch(%a_view: !hal.buffer_view, %batch: index, %w: tensor<16x12xi8>, %sa: f32, %sw: tensor<12xf32>) -> tensor<?x8x12xf32> {
  %a = hal.tensor.import %a_view : !hal.buffer_view -> tensor<?x8x16xi8>{%batch}
  %ae = tensor.empty(%batch) : tensor<?x8x16xf32>
  %adq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%a, %sa : tensor<?x8x16xi8>, f32) outs(%ae : tensor<?x8x16xf32>) -> tensor<?x8x16xf32>
  %we = tensor.empty(%batch) : tensor<?x16x12xi8>
  %wb = linalg.broadcast ins(%w : tensor<16x12xi8>) outs(%we : tensor<?x16x12xi8>) dimensions = [0]
  %wfe = tensor.empty(%batch) : tensor<?x16x12xf32>
  %wdq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%wb, %sw : tensor<?x16x12xi8>, tensor<12xf32>) outs(%wfe : tensor<?x16x12xf32>) -> tensor<?x16x12xf32>
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty(%batch) : tensor<?x8x12xf32>
  %init = linalg.fill ins(%zero : f32) outs(%ce : tensor<?x8x12xf32>) -> tensor<?x8x12xf32>
  %result = linalg.batch_matmul ins(%adq, %wdq : tensor<?x8x16xf32>, tensor<?x16x12xf32>) outs(%init : tensor<?x8x12xf32>) -> tensor<?x8x12xf32>
  util.return %result : tensor<?x8x12xf32>
}

// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(util.func(iree-global-opt-fold-broadcast-into-batch-matmul))" %s | FileCheck %s

// A weight broadcast along the batch dim folds into the RHS map.
util.func public @broadcast_rhs(%act: tensor<4x10x10xf32>, %weight: tensor<10x10xf32>) -> tensor<4x10x10xf32> {
  %init_broadcast = tensor.empty() : tensor<4x10x10xf32>
  %broadcast = linalg.broadcast ins(%weight : tensor<10x10xf32>)
                                outs(%init_broadcast : tensor<4x10x10xf32>) dimensions = [0]
  %init_out = tensor.empty() : tensor<4x10x10xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x10x10xf32>) -> tensor<4x10x10xf32>
  %result = linalg.batch_matmul ins(%act, %broadcast : tensor<4x10x10xf32>, tensor<4x10x10xf32>)
                                outs(%fill : tensor<4x10x10xf32>) -> tensor<4x10x10xf32>
  util.return %result : tensor<4x10x10xf32>
}
// CHECK-LABEL: @broadcast_rhs
//  CHECK-SAME:   %[[ACT:[a-zA-Z0-9]+]]: tensor<4x10x10xf32>, %[[WEIGHT:[a-zA-Z0-9]+]]: tensor<10x10xf32>
//   CHECK-NOT:   linalg.broadcast
//       CHECK:   linalg.batch_matmul
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     ins(%[[ACT]], %[[WEIGHT]] : tensor<4x10x10xf32>, tensor<10x10xf32>)

// -----

// Dynamic batch and M dims need no shape computation.
util.func public @broadcast_rhs_dynamic(%act: tensor<?x?x10xf32>, %weight: tensor<10x10xf32>) -> tensor<?x?x10xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %batch = tensor.dim %act, %c0 : tensor<?x?x10xf32>
  %m = tensor.dim %act, %c1 : tensor<?x?x10xf32>
  %init_broadcast = tensor.empty(%batch) : tensor<?x10x10xf32>
  %broadcast = linalg.broadcast ins(%weight : tensor<10x10xf32>)
                                outs(%init_broadcast : tensor<?x10x10xf32>) dimensions = [0]
  %init_out = tensor.empty(%batch, %m) : tensor<?x?x10xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<?x?x10xf32>) -> tensor<?x?x10xf32>
  %result = linalg.batch_matmul ins(%act, %broadcast : tensor<?x?x10xf32>, tensor<?x10x10xf32>)
                                outs(%fill : tensor<?x?x10xf32>) -> tensor<?x?x10xf32>
  util.return %result : tensor<?x?x10xf32>
}
// CHECK-LABEL: @broadcast_rhs_dynamic
//  CHECK-SAME:   %[[ACT:[a-zA-Z0-9]+]]: tensor<?x?x10xf32>, %[[WEIGHT:[a-zA-Z0-9]+]]: tensor<10x10xf32>
//   CHECK-NOT:   linalg.broadcast
//       CHECK:   linalg.batch_matmul
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     ins(%[[ACT]], %[[WEIGHT]] : tensor<?x?x10xf32>, tensor<10x10xf32>)

// -----

// The generic form of a broadcast (after generalization) folds the same way.
util.func public @generic_broadcast_rhs(%act: tensor<4x10x10xf32>, %weight: tensor<10x10xf32>) -> tensor<4x10x10xf32> {
  %init_broadcast = tensor.empty() : tensor<4x10x10xf32>
  %broadcast = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%weight : tensor<10x10xf32>) outs(%init_broadcast : tensor<4x10x10xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x10x10xf32>
  %init_out = tensor.empty() : tensor<4x10x10xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x10x10xf32>) -> tensor<4x10x10xf32>
  %result = linalg.batch_matmul ins(%act, %broadcast : tensor<4x10x10xf32>, tensor<4x10x10xf32>)
                                outs(%fill : tensor<4x10x10xf32>) -> tensor<4x10x10xf32>
  util.return %result : tensor<4x10x10xf32>
}
// CHECK-LABEL: @generic_broadcast_rhs
//  CHECK-SAME:   %[[ACT:[a-zA-Z0-9]+]]: tensor<4x10x10xf32>, %[[WEIGHT:[a-zA-Z0-9]+]]: tensor<10x10xf32>
//   CHECK-NOT:   linalg.generic
//       CHECK:   linalg.batch_matmul
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     ins(%[[ACT]], %[[WEIGHT]] : tensor<4x10x10xf32>, tensor<10x10xf32>)

// -----

// A broadcast on the LHS folds into the LHS map.
util.func public @broadcast_lhs(%lhs: tensor<10x8xf32>, %rhs: tensor<4x8x6xf32>) -> tensor<4x10x6xf32> {
  %init_broadcast = tensor.empty() : tensor<4x10x8xf32>
  %broadcast = linalg.broadcast ins(%lhs : tensor<10x8xf32>)
                                outs(%init_broadcast : tensor<4x10x8xf32>) dimensions = [0]
  %init_out = tensor.empty() : tensor<4x10x6xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x10x6xf32>) -> tensor<4x10x6xf32>
  %result = linalg.batch_matmul ins(%broadcast, %rhs : tensor<4x10x8xf32>, tensor<4x8x6xf32>)
                                outs(%fill : tensor<4x10x6xf32>) -> tensor<4x10x6xf32>
  util.return %result : tensor<4x10x6xf32>
}
// CHECK-LABEL: @broadcast_lhs
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<10x8xf32>, %[[RHS:[a-zA-Z0-9]+]]: tensor<4x8x6xf32>
//   CHECK-NOT:   linalg.broadcast
//       CHECK:   linalg.batch_matmul
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     ins(%[[LHS]], %[[RHS]] : tensor<10x8xf32>, tensor<4x8x6xf32>)

// -----

// A transposed LHS does not matter: the RHS broadcast still folds.
util.func public @broadcast_rhs_transpose_a(%act: tensor<4x10x8xf32>, %weight: tensor<10x6xf32>) -> tensor<4x8x6xf32> {
  %init_broadcast = tensor.empty() : tensor<4x10x6xf32>
  %broadcast = linalg.broadcast ins(%weight : tensor<10x6xf32>)
                                outs(%init_broadcast : tensor<4x10x6xf32>) dimensions = [0]
  %init_out = tensor.empty() : tensor<4x8x6xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x8x6xf32>) -> tensor<4x8x6xf32>
  %result = linalg.batch_matmul
      indexing_maps = [affine_map<(b, m, n, k) -> (b, k, m)>,
                       affine_map<(b, m, n, k) -> (b, k, n)>,
                       affine_map<(b, m, n, k) -> (b, m, n)>]
      ins(%act, %broadcast : tensor<4x10x8xf32>, tensor<4x10x6xf32>)
      outs(%fill : tensor<4x8x6xf32>) -> tensor<4x8x6xf32>
  util.return %result : tensor<4x8x6xf32>
}
// CHECK-LABEL: @broadcast_rhs_transpose_a
//  CHECK-SAME:   %[[ACT:[a-zA-Z0-9]+]]: tensor<4x10x8xf32>, %[[WEIGHT:[a-zA-Z0-9]+]]: tensor<10x6xf32>
//   CHECK-NOT:   linalg.broadcast
//       CHECK:   linalg.batch_matmul
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     ins(%[[ACT]], %[[WEIGHT]] : tensor<4x10x8xf32>, tensor<10x6xf32>)

// -----

// A broadcast [N, K] weight cannot be expressed as a batch_matmul broadcast
// map, so the contraction is carried by a generic with the folded map.
util.func public @broadcast_rhs_transpose_b(%act: tensor<4x8x10xf32>, %weight: tensor<6x10xf32>) -> tensor<4x8x6xf32> {
  %init_broadcast = tensor.empty() : tensor<4x6x10xf32>
  %broadcast = linalg.broadcast ins(%weight : tensor<6x10xf32>)
                                outs(%init_broadcast : tensor<4x6x10xf32>) dimensions = [0]
  %init_out = tensor.empty() : tensor<4x8x6xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x8x6xf32>) -> tensor<4x8x6xf32>
  %result = linalg.batch_matmul
      indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>,
                       affine_map<(b, m, n, k) -> (b, n, k)>,
                       affine_map<(b, m, n, k) -> (b, m, n)>]
      ins(%act, %broadcast : tensor<4x8x10xf32>, tensor<4x6x10xf32>)
      outs(%fill : tensor<4x8x6xf32>) -> tensor<4x8x6xf32>
  util.return %result : tensor<4x8x6xf32>
}
// CHECK-LABEL: @broadcast_rhs_transpose_b
//  CHECK-SAME:   %[[ACT:[a-zA-Z0-9]+]]: tensor<4x8x10xf32>, %[[WEIGHT:[a-zA-Z0-9]+]]: tensor<6x10xf32>
//   CHECK-NOT:   linalg.broadcast
//   CHECK-NOT:   linalg.batch_matmul
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "reduction"]
//  CHECK-SAME:     ins(%[[ACT]], %[[WEIGHT]] : tensor<4x8x10xf32>, tensor<6x10xf32>)
//       CHECK:     arith.mulf
//       CHECK:     arith.addf

// -----

// A generic that broadcasts and transposes (transpose propagation fuses the
// two) is handled by composing its map: here the weight is [N, K] again.
util.func public @generic_broadcast_transpose_rhs(%act: tensor<4x10x8xf32>, %weight: tensor<6x8xf32>) -> tensor<4x10x6xf32> {
  %init_broadcast = tensor.empty() : tensor<4x8x6xf32>
  %broadcast_transpose = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%weight : tensor<6x8xf32>) outs(%init_broadcast : tensor<4x8x6xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x8x6xf32>
  %init_out = tensor.empty() : tensor<4x10x6xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x10x6xf32>) -> tensor<4x10x6xf32>
  %result = linalg.batch_matmul ins(%act, %broadcast_transpose : tensor<4x10x8xf32>, tensor<4x8x6xf32>)
                                outs(%fill : tensor<4x10x6xf32>) -> tensor<4x10x6xf32>
  util.return %result : tensor<4x10x6xf32>
}
// CHECK-LABEL: @generic_broadcast_transpose_rhs
//  CHECK-SAME:   %[[ACT:[a-zA-Z0-9]+]]: tensor<4x10x8xf32>, %[[WEIGHT:[a-zA-Z0-9]+]]: tensor<6x8xf32>
//   CHECK-NOT:   linalg.batch_matmul
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "reduction"]
//  CHECK-SAME:     ins(%[[ACT]], %[[WEIGHT]] : tensor<4x10x8xf32>, tensor<6x8xf32>)
//       CHECK:     arith.mulf
//       CHECK:     arith.addf

// -----

// A broadcast that keeps the batch dim but drops K would no longer read a
// contraction operand, so nothing changes.
util.func public @broadcast_non_batch_dim(%act: tensor<4x10x10xf32>, %weight: tensor<4x10xf32>) -> tensor<4x10x10xf32> {
  %init_broadcast = tensor.empty() : tensor<4x10x10xf32>
  %broadcast = linalg.broadcast ins(%weight : tensor<4x10xf32>)
                                outs(%init_broadcast : tensor<4x10x10xf32>) dimensions = [1]
  %init_out = tensor.empty() : tensor<4x10x10xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x10x10xf32>) -> tensor<4x10x10xf32>
  %result = linalg.batch_matmul ins(%act, %broadcast : tensor<4x10x10xf32>, tensor<4x10x10xf32>)
                                outs(%fill : tensor<4x10x10xf32>) -> tensor<4x10x10xf32>
  util.return %result : tensor<4x10x10xf32>
}
// CHECK-LABEL: @broadcast_non_batch_dim
// CHECK:         linalg.broadcast
// CHECK:         linalg.batch_matmul ins(
// CHECK-NOT:     indexing_maps

// -----

// The broadcast stays for its other users; the batch_matmul still folds it.
util.func public @broadcast_with_other_user(%act: tensor<4x10x10xf32>, %weight: tensor<10x10xf32>) -> (tensor<4x10x10xf32>, tensor<4x10x10xf32>) {
  %init_broadcast = tensor.empty() : tensor<4x10x10xf32>
  %broadcast = linalg.broadcast ins(%weight : tensor<10x10xf32>)
                                outs(%init_broadcast : tensor<4x10x10xf32>) dimensions = [0]
  %init_out = tensor.empty() : tensor<4x10x10xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x10x10xf32>) -> tensor<4x10x10xf32>
  %result = linalg.batch_matmul ins(%act, %broadcast : tensor<4x10x10xf32>, tensor<4x10x10xf32>)
                                outs(%fill : tensor<4x10x10xf32>) -> tensor<4x10x10xf32>
  util.return %result, %broadcast : tensor<4x10x10xf32>, tensor<4x10x10xf32>
}
// CHECK-LABEL: @broadcast_with_other_user
//  CHECK-SAME:   %[[ACT:[a-zA-Z0-9]+]]: tensor<4x10x10xf32>, %[[WEIGHT:[a-zA-Z0-9]+]]: tensor<10x10xf32>
//       CHECK:   %[[BROADCAST:.+]] = linalg.broadcast ins(%[[WEIGHT]]
//       CHECK:   linalg.batch_matmul
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     ins(%[[ACT]], %[[WEIGHT]] : tensor<4x10x10xf32>, tensor<10x10xf32>)
//       CHECK:   util.return %{{.+}}, %[[BROADCAST]]

// -----

// A weight broadcast under a dequantize with a per-channel scale: the
// dequantize is rebuilt at the weight's rank and the broadcast folds into the
// RHS map, so the weights are dequantized once rather than once per batch.
util.func public @dequantize_broadcast_rhs(%act: tensor<2x8x256xf32>, %weight: tensor<256x1536xi8>, %scale: tensor<1536xf32>) -> tensor<2x8x1536xf32> {
  %init_broadcast = tensor.empty() : tensor<2x256x1536xi8>
  %broadcast = linalg.broadcast ins(%weight : tensor<256x1536xi8>)
                                outs(%init_broadcast : tensor<2x256x1536xi8>) dimensions = [0]
  %init_dq = tensor.empty() : tensor<2x256x1536xf32>
  %dq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%broadcast, %scale : tensor<2x256x1536xi8>, tensor<1536xf32>)
      outs(%init_dq : tensor<2x256x1536xf32>) -> tensor<2x256x1536xf32>
  %init_out = tensor.empty() : tensor<2x8x1536xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<2x8x1536xf32>) -> tensor<2x8x1536xf32>
  %result = linalg.batch_matmul ins(%act, %dq : tensor<2x8x256xf32>, tensor<2x256x1536xf32>)
                                outs(%fill : tensor<2x8x1536xf32>) -> tensor<2x8x1536xf32>
  util.return %result : tensor<2x8x1536xf32>
}
// CHECK-LABEL: @dequantize_broadcast_rhs(
//  CHECK-SAME:   %[[ACT:[a-zA-Z0-9]+]]: tensor<2x8x256xf32>, %[[WEIGHT:[a-zA-Z0-9]+]]: tensor<256x1536xi8>, %[[SCALE:[a-zA-Z0-9]+]]: tensor<1536xf32>
//   CHECK-NOT:   linalg.broadcast
//       CHECK:   %[[INIT:.+]] = tensor.empty() : tensor<256x1536xf32>
//       CHECK:   %[[DQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>]>
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]] : tensor<256x1536xi8>, tensor<1536xf32>)
//  CHECK-SAME:     outs(%[[INIT]] : tensor<256x1536xf32>)
//   CHECK-NOT:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.batch_matmul
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     ins(%[[ACT]], %[[DQ]] : tensor<2x8x256xf32>, tensor<256x1536xf32>)

// -----

// A per-channel zero point is rebased onto the weight's dims like the scale.
util.func public @dequantize_broadcast_rhs_asymmetric(%act: tensor<4x8x16xf32>, %weight: tensor<16x12xi8>, %scale: tensor<12xf32>, %zp: tensor<12xi8>) -> tensor<4x8x12xf32> {
  %init_broadcast = tensor.empty() : tensor<4x16x12xi8>
  %broadcast = linalg.broadcast ins(%weight : tensor<16x12xi8>)
                                outs(%init_broadcast : tensor<4x16x12xi8>) dimensions = [0]
  %init_dq = tensor.empty() : tensor<4x16x12xf32>
  %dq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d2)>,
                        affine_map<(d0, d1, d2) -> (d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%broadcast, %scale, %zp : tensor<4x16x12xi8>, tensor<12xf32>, tensor<12xi8>)
      outs(%init_dq : tensor<4x16x12xf32>) -> tensor<4x16x12xf32>
  %init_out = tensor.empty() : tensor<4x8x12xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  %result = linalg.batch_matmul ins(%act, %dq : tensor<4x8x16xf32>, tensor<4x16x12xf32>)
                                outs(%fill : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  util.return %result : tensor<4x8x12xf32>
}
// CHECK-LABEL: @dequantize_broadcast_rhs_asymmetric(
//  CHECK-SAME:   %[[ACT:[a-zA-Z0-9]+]]: tensor<4x8x16xf32>, %[[WEIGHT:[a-zA-Z0-9]+]]: tensor<16x12xi8>, %[[SCALE:[a-zA-Z0-9]+]]: tensor<12xf32>, %[[ZP:[a-zA-Z0-9]+]]: tensor<12xi8>
//   CHECK-NOT:   linalg.broadcast
//       CHECK:   %[[DQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>]>
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]], %[[ZP]] : tensor<16x12xi8>, tensor<12xf32>, tensor<12xi8>)
//       CHECK:   linalg.batch_matmul
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     ins(%[[ACT]], %[[DQ]] : tensor<4x8x16xf32>, tensor<16x12xf32>)

// -----

// Per-tensor scalar parameters, unsigned storage and zero point, and clamp
// bounds are carried over verbatim.
util.func public @dequantize_broadcast_rhs_per_tensor_unsigned(%act: tensor<4x8x16xf32>, %weight: tensor<16x12xi8>, %scale: f32, %zp: i8) -> tensor<4x8x12xf32> {
  %init_broadcast = tensor.empty() : tensor<4x16x12xi8>
  %broadcast = linalg.broadcast ins(%weight : tensor<16x12xi8>)
                                outs(%init_broadcast : tensor<4x16x12xi8>) dimensions = [0]
  %init_dq = tensor.empty() : tensor<4x16x12xf32>
  %dq = iree_linalg_ext.dequantize_affine
      <input_unsigned, zp_unsigned, quant_min = 0, quant_max = 255,
       indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%broadcast, %scale, %zp : tensor<4x16x12xi8>, f32, i8)
      outs(%init_dq : tensor<4x16x12xf32>) -> tensor<4x16x12xf32>
  %init_out = tensor.empty() : tensor<4x8x12xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  %result = linalg.batch_matmul ins(%act, %dq : tensor<4x8x16xf32>, tensor<4x16x12xf32>)
                                outs(%fill : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  util.return %result : tensor<4x8x12xf32>
}
// CHECK-LABEL: @dequantize_broadcast_rhs_per_tensor_unsigned(
//  CHECK-SAME:   %[[ACT:[a-zA-Z0-9]+]]: tensor<4x8x16xf32>, %[[WEIGHT:[a-zA-Z0-9]+]]: tensor<16x12xi8>, %[[SCALE:[a-zA-Z0-9]+]]: f32, %[[ZP:[a-zA-Z0-9]+]]: i8
//   CHECK-NOT:   linalg.broadcast
//       CHECK:   %[[DQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], quant_min = 0, quant_max = 255, input_unsigned, zp_unsigned>
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]], %[[ZP]] : tensor<16x12xi8>, f32, i8)
//       CHECK:   linalg.batch_matmul
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     ins(%[[ACT]], %[[DQ]] : tensor<4x8x16xf32>, tensor<16x12xf32>)

// -----

// A transpose folded into the dequantize's output map composes with the
// broadcast: the [N, K] weight is read through (d2, d3), which the named op
// rejects for a batch-broadcast RHS, so the contraction becomes a generic.
util.func public @dequantize_transposed_broadcast_rhs(%act: tensor<4x8x16xf32>, %weight: tensor<12x16xi8>, %scale: tensor<12xf32>) -> tensor<4x8x12xf32> {
  %init_broadcast = tensor.empty() : tensor<4x12x16xi8>
  %broadcast = linalg.broadcast ins(%weight : tensor<12x16xi8>)
                                outs(%init_broadcast : tensor<4x12x16xi8>) dimensions = [0]
  %init_dq = tensor.empty() : tensor<4x16x12xf32>
  %dq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d2, d1)>]>
      ins(%broadcast, %scale : tensor<4x12x16xi8>, tensor<12xf32>)
      outs(%init_dq : tensor<4x16x12xf32>) -> tensor<4x16x12xf32>
  %init_out = tensor.empty() : tensor<4x8x12xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  %result = linalg.batch_matmul ins(%act, %dq : tensor<4x8x16xf32>, tensor<4x16x12xf32>)
                                outs(%fill : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  util.return %result : tensor<4x8x12xf32>
}
// CHECK-LABEL: @dequantize_transposed_broadcast_rhs(
//  CHECK-SAME:   %[[ACT:[a-zA-Z0-9]+]]: tensor<4x8x16xf32>, %[[WEIGHT:[a-zA-Z0-9]+]]: tensor<12x16xi8>, %[[SCALE:[a-zA-Z0-9]+]]: tensor<12xf32>
//   CHECK-NOT:   linalg.broadcast
//       CHECK:   %[[DQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>]>
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]] : tensor<12x16xi8>, tensor<12xf32>)
//  CHECK-SAME:     -> tensor<12x16xf32>
//   CHECK-NOT:   linalg.batch_matmul
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     ins(%[[ACT]], %[[DQ]] : tensor<4x8x16xf32>, tensor<12x16xf32>)

// -----

// A transposed input map composes the same way.
util.func public @dequantize_transposed_input_broadcast_rhs(%act: tensor<4x8x16xf32>, %weight: tensor<12x16xi8>, %scale: tensor<12xf32>) -> tensor<4x8x12xf32> {
  %init_broadcast = tensor.empty() : tensor<4x12x16xi8>
  %broadcast = linalg.broadcast ins(%weight : tensor<12x16xi8>)
                                outs(%init_broadcast : tensor<4x12x16xi8>) dimensions = [0]
  %init_dq = tensor.empty() : tensor<4x16x12xf32>
  %dq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>,
                        affine_map<(d0, d1, d2) -> (d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%broadcast, %scale : tensor<4x12x16xi8>, tensor<12xf32>)
      outs(%init_dq : tensor<4x16x12xf32>) -> tensor<4x16x12xf32>
  %init_out = tensor.empty() : tensor<4x8x12xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  %result = linalg.batch_matmul ins(%act, %dq : tensor<4x8x16xf32>, tensor<4x16x12xf32>)
                                outs(%fill : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  util.return %result : tensor<4x8x12xf32>
}
// CHECK-LABEL: @dequantize_transposed_input_broadcast_rhs(
//  CHECK-SAME:   %[[ACT:[a-zA-Z0-9]+]]: tensor<4x8x16xf32>, %[[WEIGHT:[a-zA-Z0-9]+]]: tensor<12x16xi8>, %[[SCALE:[a-zA-Z0-9]+]]: tensor<12xf32>
//   CHECK-NOT:   linalg.broadcast
//       CHECK:   %[[DQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>]>
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]] : tensor<12x16xi8>, tensor<12xf32>)
//  CHECK-SAME:     -> tensor<12x16xf32>
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     ins(%[[ACT]], %[[DQ]] : tensor<4x8x16xf32>, tensor<12x16xf32>)

// -----

// Dynamic batch and M dims: the folded dequantize has the weight's shape and
// needs no batch extent.
util.func public @dequantize_broadcast_rhs_dynamic(%act: tensor<?x?x16xf32>, %weight: tensor<16x12xi8>, %scale: tensor<12xf32>) -> tensor<?x?x12xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %batch = tensor.dim %act, %c0 : tensor<?x?x16xf32>
  %m = tensor.dim %act, %c1 : tensor<?x?x16xf32>
  %init_broadcast = tensor.empty(%batch) : tensor<?x16x12xi8>
  %broadcast = linalg.broadcast ins(%weight : tensor<16x12xi8>)
                                outs(%init_broadcast : tensor<?x16x12xi8>) dimensions = [0]
  %init_dq = tensor.empty(%batch) : tensor<?x16x12xf32>
  %dq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%broadcast, %scale : tensor<?x16x12xi8>, tensor<12xf32>)
      outs(%init_dq : tensor<?x16x12xf32>) -> tensor<?x16x12xf32>
  %init_out = tensor.empty(%batch, %m) : tensor<?x?x12xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<?x?x12xf32>) -> tensor<?x?x12xf32>
  %result = linalg.batch_matmul ins(%act, %dq : tensor<?x?x16xf32>, tensor<?x16x12xf32>)
                                outs(%fill : tensor<?x?x12xf32>) -> tensor<?x?x12xf32>
  util.return %result : tensor<?x?x12xf32>
}
// CHECK-LABEL: @dequantize_broadcast_rhs_dynamic(
//  CHECK-SAME:   %[[ACT:[a-zA-Z0-9]+]]: tensor<?x?x16xf32>, %[[WEIGHT:[a-zA-Z0-9]+]]: tensor<16x12xi8>, %[[SCALE:[a-zA-Z0-9]+]]: tensor<12xf32>
//   CHECK-NOT:   linalg.broadcast
//       CHECK:   %[[INIT:.+]] = tensor.empty() : tensor<16x12xf32>
//       CHECK:   %[[DQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]] : tensor<16x12xi8>, tensor<12xf32>)
//  CHECK-SAME:     outs(%[[INIT]] : tensor<16x12xf32>)
//       CHECK:   linalg.batch_matmul
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     ins(%[[ACT]], %[[DQ]] : tensor<?x?x16xf32>, tensor<16x12xf32>)

// -----

// A dynamic weight dim: the folded dequantize takes its extent from the
// weight itself.
util.func public @dequantize_broadcast_rhs_dynamic_weight(%act: tensor<4x8x16xf32>, %weight: tensor<16x?xi8>, %scale: f32) -> tensor<4x8x?xf32> {
  %c1 = arith.constant 1 : index
  %n = tensor.dim %weight, %c1 : tensor<16x?xi8>
  %init_broadcast = tensor.empty(%n) : tensor<4x16x?xi8>
  %broadcast = linalg.broadcast ins(%weight : tensor<16x?xi8>)
                                outs(%init_broadcast : tensor<4x16x?xi8>) dimensions = [0]
  %init_dq = tensor.empty(%n) : tensor<4x16x?xf32>
  %dq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%broadcast, %scale : tensor<4x16x?xi8>, f32)
      outs(%init_dq : tensor<4x16x?xf32>) -> tensor<4x16x?xf32>
  %init_out = tensor.empty(%n) : tensor<4x8x?xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x8x?xf32>) -> tensor<4x8x?xf32>
  %result = linalg.batch_matmul ins(%act, %dq : tensor<4x8x16xf32>, tensor<4x16x?xf32>)
                                outs(%fill : tensor<4x8x?xf32>) -> tensor<4x8x?xf32>
  util.return %result : tensor<4x8x?xf32>
}
// CHECK-LABEL: @dequantize_broadcast_rhs_dynamic_weight(
//  CHECK-SAME:   %[[ACT:[a-zA-Z0-9]+]]: tensor<4x8x16xf32>, %[[WEIGHT:[a-zA-Z0-9]+]]: tensor<16x?xi8>, %[[SCALE:[a-zA-Z0-9]+]]: f32
//   CHECK-NOT:   linalg.broadcast
//       CHECK:   %[[INIT:.+]] = tensor.empty(%{{.+}}) : tensor<16x?xf32>
//       CHECK:   %[[DQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]] : tensor<16x?xi8>, f32)
//  CHECK-SAME:     outs(%[[INIT]] : tensor<16x?xf32>)
//       CHECK:   linalg.batch_matmul
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     ins(%[[ACT]], %[[DQ]] : tensor<4x8x16xf32>, tensor<16x?xf32>)

// -----

// A broadcast LHS under a dequantize folds into the LHS map.
util.func public @dequantize_broadcast_lhs(%lhs: tensor<10x8xi8>, %scale: f32, %rhs: tensor<4x8x6xf32>) -> tensor<4x10x6xf32> {
  %init_broadcast = tensor.empty() : tensor<4x10x8xi8>
  %broadcast = linalg.broadcast ins(%lhs : tensor<10x8xi8>)
                                outs(%init_broadcast : tensor<4x10x8xi8>) dimensions = [0]
  %init_dq = tensor.empty() : tensor<4x10x8xf32>
  %dq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%broadcast, %scale : tensor<4x10x8xi8>, f32)
      outs(%init_dq : tensor<4x10x8xf32>) -> tensor<4x10x8xf32>
  %init_out = tensor.empty() : tensor<4x10x6xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x10x6xf32>) -> tensor<4x10x6xf32>
  %result = linalg.batch_matmul ins(%dq, %rhs : tensor<4x10x8xf32>, tensor<4x8x6xf32>)
                                outs(%fill : tensor<4x10x6xf32>) -> tensor<4x10x6xf32>
  util.return %result : tensor<4x10x6xf32>
}
// CHECK-LABEL: @dequantize_broadcast_lhs(
//  CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<10x8xi8>, %[[SCALE:[a-zA-Z0-9]+]]: f32, %[[RHS:[a-zA-Z0-9]+]]: tensor<4x8x6xf32>
//   CHECK-NOT:   linalg.broadcast
//       CHECK:   %[[DQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]>
//  CHECK-SAME:     ins(%[[LHS]], %[[SCALE]] : tensor<10x8xi8>, f32)
//       CHECK:   linalg.batch_matmul
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     ins(%[[DQ]], %[[RHS]] : tensor<10x8xf32>, tensor<4x8x6xf32>)

// -----

// A broadcasting generic under the dequantize folds the same way.
util.func public @dequantize_generic_broadcast_rhs(%act: tensor<4x8x16xf32>, %weight: tensor<16x12xi8>, %scale: tensor<12xf32>) -> tensor<4x8x12xf32> {
  %init_broadcast = tensor.empty() : tensor<4x16x12xi8>
  %broadcast = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%weight : tensor<16x12xi8>) outs(%init_broadcast : tensor<4x16x12xi8>) {
  ^bb0(%in: i8, %out: i8):
    linalg.yield %in : i8
  } -> tensor<4x16x12xi8>
  %init_dq = tensor.empty() : tensor<4x16x12xf32>
  %dq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%broadcast, %scale : tensor<4x16x12xi8>, tensor<12xf32>)
      outs(%init_dq : tensor<4x16x12xf32>) -> tensor<4x16x12xf32>
  %init_out = tensor.empty() : tensor<4x8x12xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  %result = linalg.batch_matmul ins(%act, %dq : tensor<4x8x16xf32>, tensor<4x16x12xf32>)
                                outs(%fill : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  util.return %result : tensor<4x8x12xf32>
}
// CHECK-LABEL: @dequantize_generic_broadcast_rhs(
//  CHECK-SAME:   %[[ACT:[a-zA-Z0-9]+]]: tensor<4x8x16xf32>, %[[WEIGHT:[a-zA-Z0-9]+]]: tensor<16x12xi8>, %[[SCALE:[a-zA-Z0-9]+]]: tensor<12xf32>
//       CHECK:   %[[DQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]] : tensor<16x12xi8>, tensor<12xf32>)
//       CHECK:   linalg.batch_matmul
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     ins(%[[ACT]], %[[DQ]] : tensor<4x8x16xf32>, tensor<16x12xf32>)

// -----

// The integer broadcast has another user, which keeps it; the contraction
// still reads a dequantize of the unbroadcast weight.
util.func public @dequantize_broadcast_with_other_user(%act: tensor<4x8x16xf32>, %weight: tensor<16x12xi8>, %scale: tensor<12xf32>) -> (tensor<4x8x12xf32>, tensor<4x16x12xi8>) {
  %init_broadcast = tensor.empty() : tensor<4x16x12xi8>
  %broadcast = linalg.broadcast ins(%weight : tensor<16x12xi8>)
                                outs(%init_broadcast : tensor<4x16x12xi8>) dimensions = [0]
  %init_dq = tensor.empty() : tensor<4x16x12xf32>
  %dq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%broadcast, %scale : tensor<4x16x12xi8>, tensor<12xf32>)
      outs(%init_dq : tensor<4x16x12xf32>) -> tensor<4x16x12xf32>
  %init_out = tensor.empty() : tensor<4x8x12xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  %result = linalg.batch_matmul ins(%act, %dq : tensor<4x8x16xf32>, tensor<4x16x12xf32>)
                                outs(%fill : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  util.return %result, %broadcast : tensor<4x8x12xf32>, tensor<4x16x12xi8>
}
// CHECK-LABEL: @dequantize_broadcast_with_other_user(
//  CHECK-SAME:   %[[ACT:[a-zA-Z0-9]+]]: tensor<4x8x16xf32>, %[[WEIGHT:[a-zA-Z0-9]+]]: tensor<16x12xi8>, %[[SCALE:[a-zA-Z0-9]+]]: tensor<12xf32>
//       CHECK:   %[[BROADCAST:.+]] = linalg.broadcast ins(%[[WEIGHT]]
//       CHECK:   %[[DQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[WEIGHT]], %[[SCALE]] : tensor<16x12xi8>, tensor<12xf32>)
//       CHECK:   %[[RESULT:.+]] = linalg.batch_matmul
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     ins(%[[ACT]], %[[DQ]] : tensor<4x8x16xf32>, tensor<16x12xf32>)
//       CHECK:   util.return %[[RESULT]], %[[BROADCAST]]

// -----

// The dequantize has another user, so folding would add a second dequantize
// instead of shrinking this one: nothing changes.
util.func public @dequantize_broadcast_dequantize_other_user(%act: tensor<4x8x16xf32>, %weight: tensor<16x12xi8>, %scale: tensor<12xf32>) -> (tensor<4x8x12xf32>, tensor<4x16x12xf32>) {
  %init_broadcast = tensor.empty() : tensor<4x16x12xi8>
  %broadcast = linalg.broadcast ins(%weight : tensor<16x12xi8>)
                                outs(%init_broadcast : tensor<4x16x12xi8>) dimensions = [0]
  %init_dq = tensor.empty() : tensor<4x16x12xf32>
  %dq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%broadcast, %scale : tensor<4x16x12xi8>, tensor<12xf32>)
      outs(%init_dq : tensor<4x16x12xf32>) -> tensor<4x16x12xf32>
  %init_out = tensor.empty() : tensor<4x8x12xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  %result = linalg.batch_matmul ins(%act, %dq : tensor<4x8x16xf32>, tensor<4x16x12xf32>)
                                outs(%fill : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  util.return %result, %dq : tensor<4x8x12xf32>, tensor<4x16x12xf32>
}
// CHECK-LABEL: @dequantize_broadcast_dequantize_other_user(
//       CHECK:   %[[BROADCAST:.+]] = linalg.broadcast
//       CHECK:   %[[DQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[BROADCAST]], %{{.+}} : tensor<4x16x12xi8>, tensor<12xf32>)
//   CHECK-NOT:   iree_linalg_ext.dequantize_affine
//       CHECK:   %[[RESULT:.+]] = linalg.batch_matmul ins(%{{.+}}, %[[DQ]] : tensor<4x8x16xf32>, tensor<4x16x12xf32>)
//       CHECK:   util.return %[[RESULT]], %[[DQ]]

// -----

// A scale indexed by the broadcast dim varies across the batch, so the
// dequantized weights are not a broadcast of one tensor: nothing changes.
util.func public @dequantize_broadcast_batch_varying_scale(%act: tensor<4x8x16xf32>, %weight: tensor<16x12xi8>, %scale: tensor<4x12xf32>) -> tensor<4x8x12xf32> {
  %init_broadcast = tensor.empty() : tensor<4x16x12xi8>
  %broadcast = linalg.broadcast ins(%weight : tensor<16x12xi8>)
                                outs(%init_broadcast : tensor<4x16x12xi8>) dimensions = [0]
  %init_dq = tensor.empty() : tensor<4x16x12xf32>
  %dq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%broadcast, %scale : tensor<4x16x12xi8>, tensor<4x12xf32>)
      outs(%init_dq : tensor<4x16x12xf32>) -> tensor<4x16x12xf32>
  %init_out = tensor.empty() : tensor<4x8x12xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  %result = linalg.batch_matmul ins(%act, %dq : tensor<4x8x16xf32>, tensor<4x16x12xf32>)
                                outs(%fill : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  util.return %result : tensor<4x8x12xf32>
}
// CHECK-LABEL: @dequantize_broadcast_batch_varying_scale(
//       CHECK:   %[[BROADCAST:.+]] = linalg.broadcast
//       CHECK:   %[[DQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[BROADCAST]], %{{.+}} : tensor<4x16x12xi8>, tensor<4x12xf32>)
//       CHECK:   linalg.batch_matmul ins(%{{.+}}, %[[DQ]] : tensor<4x8x16xf32>, tensor<4x16x12xf32>)

// -----

// The same holds for a zero point indexed by the broadcast dim.
util.func public @dequantize_broadcast_batch_varying_zero_point(%act: tensor<4x8x16xf32>, %weight: tensor<16x12xi8>, %scale: f32, %zp: tensor<4xi8>) -> tensor<4x8x12xf32> {
  %init_broadcast = tensor.empty() : tensor<4x16x12xi8>
  %broadcast = linalg.broadcast ins(%weight : tensor<16x12xi8>)
                                outs(%init_broadcast : tensor<4x16x12xi8>) dimensions = [0]
  %init_dq = tensor.empty() : tensor<4x16x12xf32>
  %dq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> (d0)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%broadcast, %scale, %zp : tensor<4x16x12xi8>, f32, tensor<4xi8>)
      outs(%init_dq : tensor<4x16x12xf32>) -> tensor<4x16x12xf32>
  %init_out = tensor.empty() : tensor<4x8x12xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  %result = linalg.batch_matmul ins(%act, %dq : tensor<4x8x16xf32>, tensor<4x16x12xf32>)
                                outs(%fill : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  util.return %result : tensor<4x8x12xf32>
}
// CHECK-LABEL: @dequantize_broadcast_batch_varying_zero_point(
//       CHECK:   %[[BROADCAST:.+]] = linalg.broadcast
//       CHECK:   %[[DQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     ins(%[[BROADCAST]], %{{.+}}, %{{.+}} : tensor<4x16x12xi8>, f32, tensor<4xi8>)
//       CHECK:   linalg.batch_matmul ins(%{{.+}}, %[[DQ]] : tensor<4x8x16xf32>, tensor<4x16x12xf32>)

// -----

// A contraction that is already a generic folds the same way and stays a
// generic.
util.func public @generic_contraction_broadcast_rhs(%act: tensor<4x8x16xf32>, %weight: tensor<16x12xf32>) -> tensor<4x8x12xf32> {
  %init_broadcast = tensor.empty() : tensor<4x16x12xf32>
  %broadcast = linalg.broadcast ins(%weight : tensor<16x12xf32>)
                                outs(%init_broadcast : tensor<4x16x12xf32>) dimensions = [0]
  %init_out = tensor.empty() : tensor<4x8x12xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%act, %broadcast : tensor<4x8x16xf32>, tensor<4x16x12xf32>)
      outs(%fill : tensor<4x8x12xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<4x8x12xf32>
  util.return %result : tensor<4x8x12xf32>
}
// CHECK-LABEL: @generic_contraction_broadcast_rhs(
//  CHECK-SAME:   %[[ACT:[a-zA-Z0-9]+]]: tensor<4x8x16xf32>
//  CHECK-SAME:   %[[WEIGHT:[a-zA-Z0-9]+]]: tensor<16x12xf32>
//   CHECK-NOT:   linalg.broadcast
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     ins(%[[ACT]], %[[WEIGHT]] : tensor<4x8x16xf32>, tensor<16x12xf32>)

// -----

// A generic contraction over a dequantize of a broadcast: the dequantize is
// rebuilt at the weight's rank and the broadcast folds into the RHS map.
util.func public @generic_contraction_dequantize_broadcast_rhs(%act: tensor<4x8x16xf32>, %weight: tensor<16x12xi8>, %scale: tensor<12xf32>) -> tensor<4x8x12xf32> {
  %init_broadcast = tensor.empty() : tensor<4x16x12xi8>
  %broadcast = linalg.broadcast ins(%weight : tensor<16x12xi8>)
                                outs(%init_broadcast : tensor<4x16x12xi8>) dimensions = [0]
  %init_dq = tensor.empty() : tensor<4x16x12xf32>
  %dq = iree_linalg_ext.dequantize_affine
      <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%broadcast, %scale : tensor<4x16x12xi8>, tensor<12xf32>)
      outs(%init_dq : tensor<4x16x12xf32>) -> tensor<4x16x12xf32>
  %init_out = tensor.empty() : tensor<4x8x12xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%act, %dq : tensor<4x8x16xf32>, tensor<4x16x12xf32>)
      outs(%fill : tensor<4x8x12xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<4x8x12xf32>
  util.return %result : tensor<4x8x12xf32>
}
// CHECK-LABEL: @generic_contraction_dequantize_broadcast_rhs(
//  CHECK-SAME:   %[[ACT:[a-zA-Z0-9]+]]: tensor<4x8x16xf32>
//  CHECK-SAME:   %[[WEIGHT:[a-zA-Z0-9]+]]: tensor<16x12xi8>
//   CHECK-NOT:   linalg.broadcast
//       CHECK:   %[[DQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:     <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>]>
//  CHECK-SAME:     ins(%[[WEIGHT]], %{{.+}} : tensor<16x12xi8>, tensor<12xf32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     ins(%[[ACT]], %[[DQ]] : tensor<4x8x16xf32>, tensor<16x12xf32>)

// -----

// A generic with the maps of a contraction but another body is not a
// contraction, so its broadcast input is left alone.
util.func public @generic_contraction_maps_other_body(%act: tensor<4x8x16xf32>, %weight: tensor<16x12xf32>) -> tensor<4x8x12xf32> {
  %init_broadcast = tensor.empty() : tensor<4x16x12xf32>
  %broadcast = linalg.broadcast ins(%weight : tensor<16x12xf32>)
                                outs(%init_broadcast : tensor<4x16x12xf32>) dimensions = [0]
  %init_out = tensor.empty() : tensor<4x8x12xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%act, %broadcast : tensor<4x8x16xf32>, tensor<4x16x12xf32>)
      outs(%fill : tensor<4x8x12xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %mul = arith.mulf %a, %b : f32
    %max = arith.maximumf %out, %mul : f32
    linalg.yield %max : f32
  } -> tensor<4x8x12xf32>
  util.return %result : tensor<4x8x12xf32>
}
// CHECK-LABEL: @generic_contraction_maps_other_body(
//       CHECK:   %[[BROADCAST:.+]] = linalg.broadcast
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%{{.+}}, %[[BROADCAST]] : tensor<4x8x16xf32>, tensor<4x16x12xf32>)

// -----

// A broadcast along the reduction dim does not fold: the operand would no
// longer read the reduction loop.
util.func public @generic_contraction_broadcast_along_reduction(%act: tensor<8x16xf32>, %weight: tensor<12xf32>) -> tensor<8x12xf32> {
  %init_broadcast = tensor.empty() : tensor<16x12xf32>
  %broadcast = linalg.broadcast ins(%weight : tensor<12xf32>)
                                outs(%init_broadcast : tensor<16x12xf32>) dimensions = [0]
  %init_out = tensor.empty() : tensor<8x12xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<8x12xf32>) -> tensor<8x12xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%act, %broadcast : tensor<8x16xf32>, tensor<16x12xf32>)
      outs(%fill : tensor<8x12xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<8x12xf32>
  util.return %result : tensor<8x12xf32>
}
// CHECK-LABEL: @generic_contraction_broadcast_along_reduction(
//       CHECK:   %[[BROADCAST:.+]] = linalg.broadcast
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%{{.+}}, %[[BROADCAST]] : tensor<8x16xf32>, tensor<16x12xf32>)

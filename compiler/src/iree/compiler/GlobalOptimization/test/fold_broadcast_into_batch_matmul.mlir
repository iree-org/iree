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

// RUN: iree-opt --split-input-file --iree-global-optimization-transformation-pipeline="data-tiling=true" --iree-global-opt-enable-early-materialization=false %s | FileCheck %s

// The global optimization data tiling path sets encodings before dispatch
// creation, which only accepts contractions with a single M dim. A folded
// broadcast batch_matmul has two, so it is collapsed to a matmul first.

// CHECK-LABEL: @broadcast_rhs(
// CHECK-NOT:     linalg.broadcast
// CHECK-NOT:     linalg.batch_matmul
// CHECK:         %[[ACT_2D:.+]] = tensor.collapse_shape %{{.+}} {{\[}}[0, 1], [2]] : tensor<4x8x16xf32> into tensor<32x16xf32>
// CHECK-DAG:     %[[LHS:.+]] = iree_encoding.set_encoding %[[ACT_2D]] : tensor<32x16xf32> -> tensor<32x16xf32, #
// CHECK-DAG:     %[[RHS:.+]] = iree_encoding.set_encoding %{{.+}} : tensor<16x12xf32> -> tensor<16x12xf32, #
// CHECK:         %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME:      ins(%[[LHS]], %[[RHS]]
// CHECK:         %[[UNSET:.+]] = iree_encoding.unset_encoding %[[MATMUL]]
// CHECK:         tensor.expand_shape %[[UNSET]] {{\[}}[0, 1], [2]] output_shape [4, 8, 12]
util.func public @broadcast_rhs(%act: tensor<4x8x16xf32>, %weight: tensor<16x12xf32>) -> tensor<4x8x12xf32> {
  %init_broadcast = tensor.empty() : tensor<4x16x12xf32>
  %broadcast = linalg.broadcast ins(%weight : tensor<16x12xf32>)
                                outs(%init_broadcast : tensor<4x16x12xf32>) dimensions = [0]
  %init_out = tensor.empty() : tensor<4x8x12xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  %result = linalg.batch_matmul ins(%act, %broadcast : tensor<4x8x16xf32>, tensor<4x16x12xf32>)
                                outs(%fill : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  util.return %result : tensor<4x8x12xf32>
}

// -----

// A broadcast [N, K] weight is carried by a generic and takes the same path.
// CHECK-LABEL: @broadcast_rhs_transpose_b(
// CHECK-NOT:     linalg.broadcast
// CHECK-NOT:     linalg.transpose
// CHECK:         %[[ACT_2D:.+]] = tensor.collapse_shape %{{.+}} {{\[}}[0, 1], [2]] : tensor<4x8x16xf32> into tensor<32x16xf32>
// CHECK-DAG:     %[[LHS:.+]] = iree_encoding.set_encoding %[[ACT_2D]] : tensor<32x16xf32> -> tensor<32x16xf32, #
// CHECK-DAG:     %[[RHS:.+]] = iree_encoding.set_encoding %{{.+}} : tensor<12x16xf32> -> tensor<12x16xf32, #
// CHECK:         %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME:      ins(%[[LHS]], %[[RHS]]
// CHECK:         %[[UNSET:.+]] = iree_encoding.unset_encoding %[[MATMUL]]
// CHECK:         tensor.expand_shape %[[UNSET]] {{\[}}[0, 1], [2]] output_shape [4, 8, 12]
util.func public @broadcast_rhs_transpose_b(%act: tensor<4x8x16xf32>, %weight: tensor<12x16xf32>) -> tensor<4x8x12xf32> {
  %init_broadcast = tensor.empty() : tensor<4x12x16xf32>
  %broadcast = linalg.broadcast ins(%weight : tensor<12x16xf32>)
                                outs(%init_broadcast : tensor<4x12x16xf32>) dimensions = [0]
  %init_out = tensor.empty() : tensor<4x8x12xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  %result = linalg.batch_matmul
      indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>,
                       affine_map<(b, m, n, k) -> (b, n, k)>,
                       affine_map<(b, m, n, k) -> (b, m, n)>]
      ins(%act, %broadcast : tensor<4x8x16xf32>, tensor<4x12x16xf32>)
      outs(%fill : tensor<4x8x12xf32>) -> tensor<4x8x12xf32>
  util.return %result : tensor<4x8x12xf32>
}

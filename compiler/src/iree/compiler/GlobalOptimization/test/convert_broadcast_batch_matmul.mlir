// RUN: iree-opt --split-input-file --mlir-print-local-scope -iree-global-opt-convert-broadcast-batch-matmul-to-matmul %s | FileCheck %s

// Test 1: Static shapes with named broadcast
// CHECK-LABEL: util.func public @static_broadcast_batch_matmul
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<4x10x10xf32>, %[[ARG1:[a-zA-Z0-9]+]]: tensor<10x10xf32>
// CHECK-DAG:     %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<4x10x10xf32>
// CHECK:         %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<4x10x10xf32>) -> tensor<4x10x10xf32>
// CHECK:         %[[COLLAPSED_ACT:.*]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1], [2]{{\]}} : tensor<4x10x10xf32> into tensor<40x10xf32>
// CHECK:         %[[COLLAPSED_OUT:.*]] = tensor.collapse_shape %[[FILL]] {{\[}}[0, 1], [2]{{\]}} : tensor<4x10x10xf32> into tensor<40x10xf32>
// CHECK:         %[[MATMUL:.*]] = linalg.matmul ins(%[[COLLAPSED_ACT]], %[[ARG1]] : tensor<40x10xf32>, tensor<10x10xf32>) outs(%[[COLLAPSED_OUT]] : tensor<40x10xf32>) -> tensor<40x10xf32>
// CHECK:         %[[EXPANDED:.*]] = tensor.expand_shape %[[MATMUL]] {{\[}}[0, 1], [2]{{\]}} output_shape [4, 10, 10] : tensor<40x10xf32> into tensor<4x10x10xf32>
// CHECK:         util.return %[[EXPANDED]] : tensor<4x10x10xf32>
util.func public @static_broadcast_batch_matmul(%act: tensor<4x10x10xf32>, %weight: tensor<10x10xf32>) -> tensor<4x10x10xf32> {
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

// -----

// Test 2: Dynamic batch dimension (M is static)
// CHECK-LABEL: util.func public @dynamic_batch_broadcast_batch_matmul
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x10x10xf32>, %[[ARG1:[a-zA-Z0-9]+]]: tensor<10x10xf32>
// CHECK-DAG:     %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x10x10xf32>
// CHECK:         %[[EMPTY:.*]] = tensor.empty(%[[DIM]]) : tensor<?x10x10xf32>
// CHECK:         %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<?x10x10xf32>) -> tensor<?x10x10xf32>
// CHECK:         %[[COLLAPSED_ACT:.*]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1], [2]{{\]}} : tensor<?x10x10xf32> into tensor<?x10xf32>
// CHECK:         %[[COLLAPSED_OUT:.*]] = tensor.collapse_shape %[[FILL]] {{\[}}[0, 1], [2]{{\]}} : tensor<?x10x10xf32> into tensor<?x10xf32>
// CHECK:         %[[MATMUL:.*]] = linalg.matmul ins(%[[COLLAPSED_ACT]], %[[ARG1]] : tensor<?x10xf32>, tensor<10x10xf32>) outs(%[[COLLAPSED_OUT]] : tensor<?x10xf32>) -> tensor<?x10xf32>
// CHECK:         %[[BATCH:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x10x10xf32>
// CHECK:         %[[EXPANDED:.*]] = tensor.expand_shape %[[MATMUL]] {{\[}}[0, 1], [2]{{\]}} output_shape [%[[BATCH]], 10, 10] : tensor<?x10xf32> into tensor<?x10x10xf32>
// CHECK:         util.return %[[EXPANDED]] : tensor<?x10x10xf32>
util.func public @dynamic_batch_broadcast_batch_matmul(%act: tensor<?x10x10xf32>, %weight: tensor<10x10xf32>) -> tensor<?x10x10xf32> {
  %c0 = arith.constant 0 : index
  %batch = tensor.dim %act, %c0 : tensor<?x10x10xf32>
  %init_broadcast = tensor.empty(%batch) : tensor<?x10x10xf32>
  %broadcast = linalg.broadcast ins(%weight : tensor<10x10xf32>)
                                outs(%init_broadcast : tensor<?x10x10xf32>) dimensions = [0]
  %init_out = tensor.empty(%batch) : tensor<?x10x10xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<?x10x10xf32>) -> tensor<?x10x10xf32>
  %result = linalg.batch_matmul ins(%act, %broadcast : tensor<?x10x10xf32>, tensor<?x10x10xf32>)
                                outs(%fill : tensor<?x10x10xf32>) -> tensor<?x10x10xf32>
  util.return %result : tensor<?x10x10xf32>
}

// -----

// Test 3: Generic broadcast form (after generalization)
// CHECK-LABEL: util.func public @generic_broadcast_batch_matmul
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x10x10xf32>, %[[ARG1:[a-zA-Z0-9]+]]: tensor<10x10xf32>
// CHECK-DAG:     %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x10x10xf32>
// CHECK:         %[[EMPTY:.*]] = tensor.empty(%[[DIM]]) : tensor<?x10x10xf32>
// CHECK:         %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<?x10x10xf32>) -> tensor<?x10x10xf32>
// CHECK:         %[[COLLAPSED_ACT:.*]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1], [2]{{\]}} : tensor<?x10x10xf32> into tensor<?x10xf32>
// CHECK:         %[[COLLAPSED_OUT:.*]] = tensor.collapse_shape %[[FILL]] {{\[}}[0, 1], [2]{{\]}} : tensor<?x10x10xf32> into tensor<?x10xf32>
// CHECK:         %[[MATMUL:.*]] = linalg.matmul ins(%[[COLLAPSED_ACT]], %[[ARG1]] : tensor<?x10xf32>, tensor<10x10xf32>) outs(%[[COLLAPSED_OUT]] : tensor<?x10xf32>) -> tensor<?x10xf32>
// CHECK:         %[[BATCH:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x10x10xf32>
// CHECK:         %[[EXPANDED:.*]] = tensor.expand_shape %[[MATMUL]] {{\[}}[0, 1], [2]{{\]}} output_shape [%[[BATCH]], 10, 10] : tensor<?x10xf32> into tensor<?x10x10xf32>
// CHECK:         util.return %[[EXPANDED]] : tensor<?x10x10xf32>
util.func public @generic_broadcast_batch_matmul(%act: tensor<?x10x10xf32>, %weight: tensor<10x10xf32>) -> tensor<?x10x10xf32> {
  %c0 = arith.constant 0 : index
  %batch = tensor.dim %act, %c0 : tensor<?x10x10xf32>
  %init_broadcast = tensor.empty(%batch) : tensor<?x10x10xf32>
  // Generic form of broadcast (as produced by GeneralizeLinalgNamedOps)
  %broadcast = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%weight : tensor<10x10xf32>) outs(%init_broadcast : tensor<?x10x10xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<?x10x10xf32>
  %init_out = tensor.empty(%batch) : tensor<?x10x10xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<?x10x10xf32>) -> tensor<?x10x10xf32>
  %result = linalg.batch_matmul ins(%act, %broadcast : tensor<?x10x10xf32>, tensor<?x10x10xf32>)
                                outs(%fill : tensor<?x10x10xf32>) -> tensor<?x10x10xf32>
  util.return %result : tensor<?x10x10xf32>
}

// -----

// Test 4: Negative test - broadcast on LHS (should NOT transform)
// CHECK-LABEL: util.func public @broadcast_on_lhs
// CHECK:         linalg.broadcast
// CHECK:         linalg.batch_matmul
// CHECK-NOT:     tensor.collapse_shape
util.func public @broadcast_on_lhs(%lhs: tensor<10x10xf32>, %rhs: tensor<4x10x10xf32>) -> tensor<4x10x10xf32> {
  %init_broadcast = tensor.empty() : tensor<4x10x10xf32>
  %broadcast = linalg.broadcast ins(%lhs : tensor<10x10xf32>)
                                outs(%init_broadcast : tensor<4x10x10xf32>) dimensions = [0]
  %init_out = tensor.empty() : tensor<4x10x10xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x10x10xf32>) -> tensor<4x10x10xf32>
  %result = linalg.batch_matmul ins(%broadcast, %rhs : tensor<4x10x10xf32>, tensor<4x10x10xf32>)
                                outs(%fill : tensor<4x10x10xf32>) -> tensor<4x10x10xf32>
  util.return %result : tensor<4x10x10xf32>
}

// -----

// Test 5: Both batch and M are dynamic (should transform with explicit output_shape)
// CHECK-LABEL: util.func public @both_batch_and_m_dynamic
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x10xf32>, %[[ARG1:[a-zA-Z0-9]+]]: tensor<10x10xf32>
// CHECK-DAG:     %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[BATCH:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x10xf32>
// CHECK:         %[[M:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x10xf32>
// CHECK:         %[[EMPTY:.*]] = tensor.empty(%[[BATCH]], %[[M]]) : tensor<?x?x10xf32>
// CHECK:         %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<?x?x10xf32>) -> tensor<?x?x10xf32>
// CHECK:         %[[COLLAPSED_ACT:.*]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1], [2]{{\]}} : tensor<?x?x10xf32> into tensor<?x10xf32>
// CHECK:         %[[COLLAPSED_OUT:.*]] = tensor.collapse_shape %[[FILL]] {{\[}}[0, 1], [2]{{\]}} : tensor<?x?x10xf32> into tensor<?x10xf32>
// CHECK:         %[[MATMUL:.*]] = linalg.matmul ins(%[[COLLAPSED_ACT]], %[[ARG1]] : tensor<?x10xf32>, tensor<10x10xf32>) outs(%[[COLLAPSED_OUT]] : tensor<?x10xf32>) -> tensor<?x10xf32>
// CHECK:         %[[DIM_BATCH:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x10xf32>
// CHECK:         %[[DIM_M:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x10xf32>
// CHECK:         %[[EXPANDED:.*]] = tensor.expand_shape %[[MATMUL]] {{\[}}[0, 1], [2]{{\]}} output_shape [%[[DIM_BATCH]], %[[DIM_M]], 10] : tensor<?x10xf32> into tensor<?x?x10xf32>
// CHECK:         util.return %[[EXPANDED]] : tensor<?x?x10xf32>
util.func public @both_batch_and_m_dynamic(%act: tensor<?x?x10xf32>, %weight: tensor<10x10xf32>) -> tensor<?x?x10xf32> {
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

// -----

// Test 6: Negative test - broadcast adds dimension other than batch (should NOT transform)
// CHECK-LABEL: util.func public @broadcast_non_batch_dim
// CHECK:         linalg.broadcast
// CHECK:         linalg.batch_matmul
// CHECK-NOT:     tensor.collapse_shape
util.func public @broadcast_non_batch_dim(%act: tensor<4x10x10xf32>, %weight: tensor<4x10xf32>) -> tensor<4x10x10xf32> {
  %init_broadcast = tensor.empty() : tensor<4x10x10xf32>
  // Broadcast adds dimension 1, not 0
  %broadcast = linalg.broadcast ins(%weight : tensor<4x10xf32>)
                                outs(%init_broadcast : tensor<4x10x10xf32>) dimensions = [1]
  %init_out = tensor.empty() : tensor<4x10x10xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x10x10xf32>) -> tensor<4x10x10xf32>
  %result = linalg.batch_matmul ins(%act, %broadcast : tensor<4x10x10xf32>, tensor<4x10x10xf32>)
                                outs(%fill : tensor<4x10x10xf32>) -> tensor<4x10x10xf32>
  util.return %result : tensor<4x10x10xf32>
}

// -----

// Test 7: Transpose B variant (RHS weight is [N, K] instead of [K, N])
// batch_matmul with indexing_maps specifying transpose_b: RHS accesses (batch, n, k)
// Activation: [4, 8, 10] = [batch, M, K]
// Weight: [6, 10] = [N, K] (transposed)
// After broadcast: [4, 6, 10] = [batch, N, K]
// Output: [4, 8, 6] = [batch, M, N]
// CHECK-LABEL: util.func public @broadcast_batch_matmul_transpose_b
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<4x8x10xf32>, %[[ARG1:[a-zA-Z0-9]+]]: tensor<6x10xf32>
// CHECK-DAG:     %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<4x8x6xf32>
// CHECK:         %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<4x8x6xf32>) -> tensor<4x8x6xf32>
// CHECK:         %[[COLLAPSED_ACT:.*]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1], [2]{{\]}} : tensor<4x8x10xf32> into tensor<32x10xf32>
// CHECK:         %[[COLLAPSED_OUT:.*]] = tensor.collapse_shape %[[FILL]] {{\[}}[0, 1], [2]{{\]}} : tensor<4x8x6xf32> into tensor<32x6xf32>
// CHECK:         %[[MATMUL:.*]] = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%[[COLLAPSED_ACT]], %[[ARG1]] : tensor<32x10xf32>, tensor<6x10xf32>) outs(%[[COLLAPSED_OUT]] : tensor<32x6xf32>) -> tensor<32x6xf32>
// CHECK:         %[[EXPANDED:.*]] = tensor.expand_shape %[[MATMUL]] {{\[}}[0, 1], [2]{{\]}} output_shape [4, 8, 6] : tensor<32x6xf32> into tensor<4x8x6xf32>
// CHECK:         util.return %[[EXPANDED]] : tensor<4x8x6xf32>
util.func public @broadcast_batch_matmul_transpose_b(%act: tensor<4x8x10xf32>, %weight: tensor<6x10xf32>) -> tensor<4x8x6xf32> {
  %init_broadcast = tensor.empty() : tensor<4x6x10xf32>
  %broadcast = linalg.broadcast ins(%weight : tensor<6x10xf32>)
                                outs(%init_broadcast : tensor<4x6x10xf32>) dimensions = [0]
  %init_out = tensor.empty() : tensor<4x8x6xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x8x6xf32>) -> tensor<4x8x6xf32>
  // batch_matmul with transpose_b indexing: RHS is (batch, n, k) instead of (batch, k, n)
  %result = linalg.batch_matmul
      indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>,
                       affine_map<(b, m, n, k) -> (b, n, k)>,
                       affine_map<(b, m, n, k) -> (b, m, n)>]
      ins(%act, %broadcast : tensor<4x8x10xf32>, tensor<4x6x10xf32>)
      outs(%fill : tensor<4x8x6xf32>) -> tensor<4x8x6xf32>
  util.return %result : tensor<4x8x6xf32>
}

// -----

// Test 8: Negative test - fused broadcast + transpose in generic (should NOT transform)
// When transpose propagation folds a transpose into the broadcast generic, the
// input map becomes (d0,d1,d2) -> (d2,d1) instead of (d0,d1,d2) -> (d1,d2).
// The pass must not treat this as a pure broadcast.
// CHECK-LABEL: util.func public @fused_broadcast_transpose_generic
// CHECK:         linalg.generic
// CHECK:         linalg.batch_matmul
// CHECK-NOT:     tensor.collapse_shape
util.func public @fused_broadcast_transpose_generic(%act: tensor<4x10x8xf32>, %weight: tensor<6x8xf32>) -> tensor<4x10x6xf32> {
  %init_broadcast = tensor.empty() : tensor<4x8x6xf32>
  // Generic that broadcasts on dim 0 AND transposes the inner dims
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

// -----

// Test 9: Transpose A variant (LHS activation is [batch, K, M] instead of [batch, M, K])
// This should NOT transform because the collapse would produce mismatched shapes:
// - LHS [batch, K, M] collapses to [batch*K, M]
// - Out [batch, M, N] collapses to [batch*M, N]
// These don't match, so transformation is invalid.
// CHECK-LABEL: util.func public @broadcast_batch_matmul_transpose_a
// CHECK:         linalg.broadcast
// CHECK:         linalg.batch_matmul
// CHECK-NOT:     tensor.collapse_shape
util.func public @broadcast_batch_matmul_transpose_a(%act: tensor<4x10x8xf32>, %weight: tensor<10x6xf32>) -> tensor<4x8x6xf32> {
  %init_broadcast = tensor.empty() : tensor<4x10x6xf32>
  %broadcast = linalg.broadcast ins(%weight : tensor<10x6xf32>)
                                outs(%init_broadcast : tensor<4x10x6xf32>) dimensions = [0]
  %init_out = tensor.empty() : tensor<4x8x6xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<4x8x6xf32>) -> tensor<4x8x6xf32>
  // batch_matmul with transpose_a indexing: LHS is (batch, k, m) instead of (batch, m, k)
  %result = linalg.batch_matmul
      indexing_maps = [affine_map<(b, m, n, k) -> (b, k, m)>,
                       affine_map<(b, m, n, k) -> (b, k, n)>,
                       affine_map<(b, m, n, k) -> (b, m, n)>]
      ins(%act, %broadcast : tensor<4x10x8xf32>, tensor<4x10x6xf32>)
      outs(%fill : tensor<4x8x6xf32>) -> tensor<4x8x6xf32>
  util.return %result : tensor<4x8x6xf32>
}

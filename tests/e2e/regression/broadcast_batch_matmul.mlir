// End-to-end correctness test for folding a batch-dim broadcast into a
// batch_matmul's indexing maps (--iree-global-opt-fold-broadcast-into-batch-matmul),
// which runs as part of the default global optimization pipeline, and for the
// collapse of the resulting contraction into a plain matmul during dispatch
// creation. Each %ref hides its broadcast behind an optimization barrier so
// the fold does not apply to it.

func.func @broadcast_batch_matmul() {
  %act = util.unfoldable_constant dense<[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                         [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]> : tensor<2x2x3xf32>
  %weight = util.unfoldable_constant dense<[[1.0, 0.0, 0.0, 1.0],
                                            [0.0, 1.0, 0.0, 1.0],
                                            [0.0, 0.0, 1.0, 1.0]]> : tensor<3x4xf32>
  %cst = arith.constant 0.0 : f32
  %init_broadcast = tensor.empty() : tensor<2x3x4xf32>
  %broadcast = linalg.broadcast ins(%weight : tensor<3x4xf32>)
                                outs(%init_broadcast : tensor<2x3x4xf32>) dimensions = [0]
  %init_out = tensor.empty() : tensor<2x2x4xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>

  %opt = linalg.batch_matmul ins(%act, %broadcast : tensor<2x2x3xf32>, tensor<2x3x4xf32>)
                             outs(%fill : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>

  %broadcast_ref = util.optimization_barrier %broadcast : tensor<2x3x4xf32>
  %ref = linalg.batch_matmul ins(%act, %broadcast_ref : tensor<2x2x3xf32>, tensor<2x3x4xf32>)
                             outs(%fill : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>

  check.expect_almost_eq(%opt, %ref) : tensor<2x2x4xf32>
  return
}

func.func @broadcast_batch_matmul_transpose_b() {
  %act = util.unfoldable_constant dense<[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                         [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]> : tensor<2x2x3xf32>
  %weight = util.unfoldable_constant dense<[[1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 1.0],
                                            [1.0, 1.0, 1.0]]> : tensor<4x3xf32>
  %cst = arith.constant 0.0 : f32
  %init_broadcast = tensor.empty() : tensor<2x4x3xf32>
  %broadcast = linalg.broadcast ins(%weight : tensor<4x3xf32>)
                                outs(%init_broadcast : tensor<2x4x3xf32>) dimensions = [0]
  %init_out = tensor.empty() : tensor<2x2x4xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>

  %opt = linalg.batch_matmul
      indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>,
                       affine_map<(b, m, n, k) -> (b, n, k)>,
                       affine_map<(b, m, n, k) -> (b, m, n)>]
      ins(%act, %broadcast : tensor<2x2x3xf32>, tensor<2x4x3xf32>)
      outs(%fill : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>

  %broadcast_ref = util.optimization_barrier %broadcast : tensor<2x4x3xf32>
  %ref = linalg.batch_matmul
      indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>,
                       affine_map<(b, m, n, k) -> (b, n, k)>,
                       affine_map<(b, m, n, k) -> (b, m, n)>]
      ins(%act, %broadcast_ref : tensor<2x2x3xf32>, tensor<2x4x3xf32>)
      outs(%fill : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>

  check.expect_almost_eq(%opt, %ref) : tensor<2x2x4xf32>
  return
}

func.func @broadcast_batch_matmul_transpose_a() {
  %act = util.unfoldable_constant dense<[[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
                                         [[7.0, 10.0], [8.0, 11.0], [9.0, 12.0]]]> : tensor<2x3x2xf32>
  %weight = util.unfoldable_constant dense<[[1.0, 0.0, 0.0, 1.0],
                                            [0.0, 1.0, 0.0, 1.0],
                                            [0.0, 0.0, 1.0, 1.0]]> : tensor<3x4xf32>
  %cst = arith.constant 0.0 : f32
  %init_broadcast = tensor.empty() : tensor<2x3x4xf32>
  %broadcast = linalg.broadcast ins(%weight : tensor<3x4xf32>)
                                outs(%init_broadcast : tensor<2x3x4xf32>) dimensions = [0]
  %init_out = tensor.empty() : tensor<2x2x4xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>

  %opt = linalg.batch_matmul
      indexing_maps = [affine_map<(b, m, n, k) -> (b, k, m)>,
                       affine_map<(b, m, n, k) -> (b, k, n)>,
                       affine_map<(b, m, n, k) -> (b, m, n)>]
      ins(%act, %broadcast : tensor<2x3x2xf32>, tensor<2x3x4xf32>)
      outs(%fill : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>

  %broadcast_ref = util.optimization_barrier %broadcast : tensor<2x3x4xf32>
  %ref = linalg.batch_matmul
      indexing_maps = [affine_map<(b, m, n, k) -> (b, k, m)>,
                       affine_map<(b, m, n, k) -> (b, k, n)>,
                       affine_map<(b, m, n, k) -> (b, m, n)>]
      ins(%act, %broadcast_ref : tensor<2x3x2xf32>, tensor<2x3x4xf32>)
      outs(%fill : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>

  check.expect_almost_eq(%opt, %ref) : tensor<2x2x4xf32>
  return
}

func.func @broadcast_batch_matmul_lhs() {
  %act = util.unfoldable_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %weight = util.unfoldable_constant dense<[[[1.0, 0.0, 0.0, 1.0],
                                             [0.0, 1.0, 0.0, 1.0],
                                             [0.0, 0.0, 1.0, 1.0]],
                                            [[2.0, 0.0, 0.0, 1.0],
                                             [0.0, 2.0, 0.0, 1.0],
                                             [0.0, 0.0, 2.0, 1.0]]]> : tensor<2x3x4xf32>
  %cst = arith.constant 0.0 : f32
  %init_broadcast = tensor.empty() : tensor<2x2x3xf32>
  %broadcast = linalg.broadcast ins(%act : tensor<2x3xf32>)
                                outs(%init_broadcast : tensor<2x2x3xf32>) dimensions = [0]
  %init_out = tensor.empty() : tensor<2x2x4xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init_out : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>

  %opt = linalg.batch_matmul ins(%broadcast, %weight : tensor<2x2x3xf32>, tensor<2x3x4xf32>)
                             outs(%fill : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>

  %broadcast_ref = util.optimization_barrier %broadcast : tensor<2x2x3xf32>
  %ref = linalg.batch_matmul ins(%broadcast_ref, %weight : tensor<2x2x3xf32>, tensor<2x3x4xf32>)
                             outs(%fill : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>

  check.expect_almost_eq(%opt, %ref) : tensor<2x2x4xf32>
  return
}

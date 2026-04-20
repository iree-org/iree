// Regression test for dynamic batch_matmul with WMMAR3 on RDNA3 (gfx1100).
//
// WMMAR3 has accumulator layout outer={8,1} which requires an expand_shape
// on the output. With dynamic shapes, tensor.dim users on the forall result
// previously blocked the ExpandDestinationForallOp pattern, causing a
// separate shared memory allocation for the output accumulator that exceeded
// the 65536-byte limit.

func.func @dynamic_batch_matmul_transposed_rhs() {
  %lhs = flow.tensor.dynamic_constant dense<1.0> : tensor<2x128x128xf16> -> tensor<2x?x128xf16>
  %rhs = flow.tensor.dynamic_constant dense<1.0> : tensor<2x128x128xf16> -> tensor<2x?x128xf16>

  %cst = arith.constant 0.0 : f32
  %c1 = arith.constant 1 : index
  %m = tensor.dim %lhs, %c1 : tensor<2x?x128xf16>
  %n = tensor.dim %rhs, %c1 : tensor<2x?x128xf16>
  %init = tensor.empty(%m, %n) : tensor<2x?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
  %observed = linalg.batch_matmul
    indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>,
                     affine_map<(b, m, n, k) -> (b, n, k)>,
                     affine_map<(b, m, n, k) -> (b, m, n)>]
    ins(%lhs, %rhs : tensor<2x?x128xf16>, tensor<2x?x128xf16>)
    outs(%fill : tensor<2x?x?xf32>) -> tensor<2x?x?xf32>

  // Each output element = sum(1.0 * 1.0, K=128) = 128.0
  %expected = flow.tensor.dynamic_constant dense<128.0> : tensor<2x128x128xf32> -> tensor<2x?x?xf32>
  check.expect_almost_eq(%observed, %expected, atol 1.0e-01) : tensor<2x?x?xf32>
  return
}

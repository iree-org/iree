// Batch matmul whose RHS is a constant [K, N] weight broadcast over the batch
// through the indexing maps, BxMxNxK = 8x512x512x512, f32.

util.global private @weight = #util.byte_pattern<63> : tensor<512x512xf32>

func.func @batch_matmul_bcast_rhs_8x512x512x512_f32(%a: tensor<8x512x512xf32>) -> tensor<8x512x512xf32> {
  %weight = util.global.load @weight : tensor<512x512xf32>
  %e1 = tensor.empty() : tensor<8x512x512xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<8x512x512xf32>) -> tensor<8x512x512xf32>
  %bmm4 = linalg.batch_matmul indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>, affine_map<(b, m, n, k) -> (k, n)>, affine_map<(b, m, n, k) -> (b, m, n)>] ins(%a, %weight : tensor<8x512x512xf32>, tensor<512x512xf32>) outs(%f3 : tensor<8x512x512xf32>) -> tensor<8x512x512xf32>
  return %bmm4 : tensor<8x512x512xf32>
}

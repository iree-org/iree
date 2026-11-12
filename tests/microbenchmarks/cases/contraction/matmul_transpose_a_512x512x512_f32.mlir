// Matmul with transposed LHS (indexing maps), MxNxK = 512x512x512, f32.

func.func @matmul_transpose_a_512x512x512_f32(%a: tensor<512x512xf32>, %b: tensor<512x512xf32>) -> tensor<512x512xf32> {
  %e1 = tensor.empty() : tensor<512x512xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<512x512xf32>) -> tensor<512x512xf32>
  %mm4 = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (k, m)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>] ins(%a, %b : tensor<512x512xf32>, tensor<512x512xf32>) outs(%f3 : tensor<512x512xf32>) -> tensor<512x512xf32>
  return %mm4 : tensor<512x512xf32>
}

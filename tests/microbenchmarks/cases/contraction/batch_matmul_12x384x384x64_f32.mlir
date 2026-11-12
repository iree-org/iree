// Batch matmul BxMxNxK = 12x384x384x64 (attention scores shape), f32.

func.func @batch_matmul_12x384x384x64_f32(%a: tensor<12x384x64xf32>, %b: tensor<12x64x384xf32>) -> tensor<12x384x384xf32> {
  %e1 = tensor.empty() : tensor<12x384x384xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<12x384x384xf32>) -> tensor<12x384x384xf32>
  %bmm4 = linalg.batch_matmul ins(%a, %b : tensor<12x384x64xf32>, tensor<12x64x384xf32>) outs(%f3 : tensor<12x384x384xf32>) -> tensor<12x384x384xf32>
  return %bmm4 : tensor<12x384x384xf32>
}

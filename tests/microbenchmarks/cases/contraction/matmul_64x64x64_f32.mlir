// Matmul MxNxK = 64x64x64, f32, both operands are function arguments.

func.func @matmul_64x64x64_f32(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %e1 = tensor.empty() : tensor<64x64xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %mm4 = linalg.matmul ins(%a, %b : tensor<64x64xf32>, tensor<64x64xf32>) outs(%f3 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %mm4 : tensor<64x64xf32>
}

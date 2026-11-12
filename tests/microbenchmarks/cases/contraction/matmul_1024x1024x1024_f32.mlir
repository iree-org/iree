// Matmul MxNxK = 1024x1024x1024, f32, both operands are function arguments.

func.func @matmul_1024x1024x1024_f32(%a: tensor<1024x1024xf32>, %b: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
  %e1 = tensor.empty() : tensor<1024x1024xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %mm4 = linalg.matmul ins(%a, %b : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%f3 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  return %mm4 : tensor<1024x1024xf32>
}

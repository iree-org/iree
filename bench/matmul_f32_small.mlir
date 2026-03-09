func.func @matmul_64x64x64_f32(%lhs: tensor<64x64xf32>, %rhs: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<64x64xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<64x64xf32>, tensor<64x64xf32>) outs(%fill : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

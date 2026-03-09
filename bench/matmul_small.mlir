// Single matmul, e2e test shape
func.func @matmul_512x128x512(%lhs: tensor<512x512xf16>, %rhs: tensor<512x128xf16>) -> tensor<512x128xf32> {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<512x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<512x128xf32>) -> tensor<512x128xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<512x512xf16>, tensor<512x128xf16>) outs(%fill : tensor<512x128xf32>) -> tensor<512x128xf32>
  return %result : tensor<512x128xf32>
}

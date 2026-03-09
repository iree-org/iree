// f32 matmul — no WMMA, uses scalar/SIMD codegen
func.func @matmul_2048x2048x2048_f32(%lhs: tensor<2048x2048xf32>, %rhs: tensor<2048x2048xf32>) -> tensor<2048x2048xf32> {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<2048x2048xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<2048x2048xf32>, tensor<2048x2048xf32>) outs(%fill : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
  return %result : tensor<2048x2048xf32>
}

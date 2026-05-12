// Matmul benchmark: f16 inputs, f32 accumulator

func.func @matmul_2048x2048x2048(%lhs: tensor<2048x2048xf16>, %rhs: tensor<2048x2048xf16>) -> tensor<2048x2048xf32> {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<2048x2048xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<2048x2048xf16>, tensor<2048x2048xf16>) outs(%fill : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
  return %result : tensor<2048x2048xf32>
}

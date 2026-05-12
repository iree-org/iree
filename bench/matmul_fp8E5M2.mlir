// Matmul benchmark: fp8 (f8E5M2) inputs, f32 accumulator
// Shapes: 2048x2048x2048, 2048x1024x4096, 4096x4096x4096

func.func @matmul_2048x2048x2048(%lhs: tensor<2048x2048xf8E5M2>, %rhs: tensor<2048x2048xf8E5M2>) -> tensor<2048x2048xf32> {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<2048x2048xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<2048x2048xf8E5M2>, tensor<2048x2048xf8E5M2>) outs(%fill : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
  return %result : tensor<2048x2048xf32>
}

func.func @matmul_2048x1024x4096(%lhs: tensor<2048x4096xf8E5M2>, %rhs: tensor<4096x1024xf8E5M2>) -> tensor<2048x1024xf32> {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<2048x1024xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<2048x1024xf32>) -> tensor<2048x1024xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<2048x4096xf8E5M2>, tensor<4096x1024xf8E5M2>) outs(%fill : tensor<2048x1024xf32>) -> tensor<2048x1024xf32>
  return %result : tensor<2048x1024xf32>
}

func.func @matmul_4096x4096x4096(%lhs: tensor<4096x4096xf8E5M2>, %rhs: tensor<4096x4096xf8E5M2>) -> tensor<4096x4096xf32> {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<4096x4096xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<4096x4096xf8E5M2>, tensor<4096x4096xf8E5M2>) outs(%fill : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  return %result : tensor<4096x4096xf32>
}

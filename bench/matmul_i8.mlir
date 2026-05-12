// Matmul benchmark: i8 inputs, i32 accumulator
// Shapes: 2048x2048x2048, 2048x1024x4096, 4096x4096x4096

func.func @matmul_2048x2048x2048(%lhs: tensor<2048x2048xi8>, %rhs: tensor<2048x2048xi8>) -> tensor<2048x2048xi32> {
  %cst = arith.constant 0 : i32
  %init = tensor.empty() : tensor<2048x2048xi32>
  %fill = linalg.fill ins(%cst : i32) outs(%init : tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<2048x2048xi8>, tensor<2048x2048xi8>) outs(%fill : tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
  return %result : tensor<2048x2048xi32>
}

func.func @matmul_2048x1024x4096(%lhs: tensor<2048x4096xi8>, %rhs: tensor<4096x1024xi8>) -> tensor<2048x1024xi32> {
  %cst = arith.constant 0 : i32
  %init = tensor.empty() : tensor<2048x1024xi32>
  %fill = linalg.fill ins(%cst : i32) outs(%init : tensor<2048x1024xi32>) -> tensor<2048x1024xi32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<2048x4096xi8>, tensor<4096x1024xi8>) outs(%fill : tensor<2048x1024xi32>) -> tensor<2048x1024xi32>
  return %result : tensor<2048x1024xi32>
}

func.func @matmul_4096x4096x4096(%lhs: tensor<4096x4096xi8>, %rhs: tensor<4096x4096xi8>) -> tensor<4096x4096xi32> {
  %cst = arith.constant 0 : i32
  %init = tensor.empty() : tensor<4096x4096xi32>
  %fill = linalg.fill ins(%cst : i32) outs(%init : tensor<4096x4096xi32>) -> tensor<4096x4096xi32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<4096x4096xi8>, tensor<4096x4096xi8>) outs(%fill : tensor<4096x4096xi32>) -> tensor<4096x4096xi32>
  return %result : tensor<4096x4096xi32>
}

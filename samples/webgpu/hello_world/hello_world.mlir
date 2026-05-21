func.func @hello_world(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {
  %sum = arith.addf %lhs, %rhs : tensor<4xf32>
  %product = arith.mulf %sum, %rhs : tensor<4xf32>
  return %product : tensor<4xf32>
}

func.func @add_2048(%lhs: tensor<2048x2048xf32>, %rhs: tensor<2048x2048xf32>) -> tensor<2048x2048xf32> {
  %result = arith.addf %lhs, %rhs : tensor<2048x2048xf32>
  return %result : tensor<2048x2048xf32>
}

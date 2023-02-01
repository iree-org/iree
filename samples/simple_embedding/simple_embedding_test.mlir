func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>
  return %0, %arg0 : tensor<4xf32>, tensor<4xf32>
}

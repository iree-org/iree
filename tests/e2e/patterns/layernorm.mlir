// y = gamma * (x-mean(x)) / rsqrt(var(x) + epsilon) + beta
// Setting gamma = 1.0 and beta = 0.0 for simplicity.
func.func @layernorm() {
  %x = util.unfoldable_constant dense<5.0> : tensor<128x384xf32>
  %c384 = util.unfoldable_constant dense<384.0> : tensor<128x1xf32>
  %sum = "tosa.reduce_sum"(%x) {axis = 1 : i64} : (tensor<128x384xf32>) -> tensor<128x1xf32>
  %r384 = "tosa.reciprocal"(%c384) : (tensor<128x1xf32>) -> tensor<128x1xf32>
  %mean = "tosa.mul"(%sum, %r384) {shift = 0 : i32} : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
  %x_sub_mean = "tosa.sub"(%x, %mean) : (tensor<128x384xf32>, tensor<128x1xf32>) -> tensor<128x384xf32>
  %square = "tosa.mul"(%x_sub_mean, %x_sub_mean) {shift = 0 : i32} : (tensor<128x384xf32>, tensor<128x384xf32>) -> tensor<128x384xf32>
  %square_sum = "tosa.reduce_sum"(%square) {axis = 1 : i64} : (tensor<128x384xf32>) -> tensor<128x1xf32>
  %variance = "tosa.mul"(%square_sum, %r384) {shift = 0 : i32} : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
  %epsilon = util.unfoldable_constant dense<9.99999996E-13> : tensor<128x1xf32>
  %var_eps = "tosa.add"(%variance, %epsilon) : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
  %rsigma = "tosa.rsqrt"(%var_eps) : (tensor<128x1xf32>) -> tensor<128x1xf32>
  %norm = "tosa.mul"(%x_sub_mean, %rsigma) {shift = 0 : i32} : (tensor<128x384xf32>, tensor<128x1xf32>) -> tensor<128x384xf32>
  check.expect_almost_eq_const(%norm, dense<0.0> : tensor<128x384xf32>) : tensor<128x384xf32>
  return
}

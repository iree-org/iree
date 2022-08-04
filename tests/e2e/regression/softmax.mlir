func.func @softmax() {
  %0 = util.unfoldable_constant dense<5.0> : tensor<12x128x128xf32>
  %red = "tosa.reduce_max"(%0) {axis = 2 : i64} : (tensor<12x128x128xf32>) -> tensor<12x128x1xf32>
  %sub = "tosa.sub"(%0, %red) : (tensor<12x128x128xf32>, tensor<12x128x1xf32>) -> tensor<12x128x128xf32>
  %exp = "tosa.exp"(%sub) : (tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
  %sum = "tosa.reduce_sum"(%exp) {axis = 2 : i64} : (tensor<12x128x128xf32>) -> tensor<12x128x1xf32>
  %rec = "tosa.reciprocal"(%sum) : (tensor<12x128x1xf32>) -> tensor<12x128x1xf32>
  %mul = "tosa.mul"(%exp, %rec) {shift = 0 : i32} : (tensor<12x128x128xf32>, tensor<12x128x1xf32>) -> tensor<12x128x128xf32>
  check.expect_almost_eq_const(%mul, dense<0.0078125> : tensor<12x128x128xf32>) : tensor<12x128x128xf32>
  return
}

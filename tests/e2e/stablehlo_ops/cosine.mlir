func.func @tensor() {
  %input = util.unfoldable_constant dense<[0.0, 1.0, 1.5, 2.0]> : tensor<4xf32>
  %result = "stablehlo.cosine"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[1.0, 0.5403, 0.0707, -0.4161]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func.func @scalar() {
  %input = util.unfoldable_constant dense<3.0> : tensor<f32>
  %result = "stablehlo.cosine"(%input) : (tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<-0.99> : tensor<f32>) : tensor<f32>
  return
}

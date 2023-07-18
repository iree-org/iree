func.func @tensor() {
  %input = util.unfoldable_constant dense<[-1.0, -2.0, 3.0, 4.0]> : tensor<4xf32>
  %result = "stablehlo.abs"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func.func @scalar() {
  %input = util.unfoldable_constant dense<-4.0> : tensor<f32>
  %result = "stablehlo.abs"(%input) : (tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<4.0> : tensor<f32>) : tensor<f32>
  return
}

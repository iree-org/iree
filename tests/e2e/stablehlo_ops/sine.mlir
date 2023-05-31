func.func @tensor() {
  %input = util.unfoldable_constant dense<[0.0, 1.0, 1.5, 2.0]> : tensor<4xf32>
  %result = "stablehlo.sine"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[0.0, 0.8415, 0.9975, 0.9093]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func.func @scalar() {
  %input = util.unfoldable_constant dense<3.0> : tensor<f32>
  %result = "stablehlo.sine"(%input) : (tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<0.14112> : tensor<f32>) : tensor<f32>
  return
}

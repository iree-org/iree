func.func @tensor() {
  %cst = stablehlo.constant dense<3.0e+00> : tensor<4xf32>
  %input = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %result = "stablehlo.power"(%input, %cst) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[1.0, 8.0, 27.0, 64.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func.func @scalar() {
  %cst = stablehlo.constant dense<2.0e+00> : tensor<f32>
  %input = util.unfoldable_constant dense<16.0> : tensor<f32>
  %result = "stablehlo.power"(%input, %cst) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<256.0> : tensor<f32>) : tensor<f32>
  return
}

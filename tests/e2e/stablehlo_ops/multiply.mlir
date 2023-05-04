func.func @multiply () {
  %c2 = util.unfoldable_constant dense<2.0> : tensor<f32>
  %res = "stablehlo.multiply"(%c2, %c2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%res, dense<4.0> : tensor<f32>) : tensor<f32>
  return
}

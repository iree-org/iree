func.func @log_plus_one() {
  %input = util.unfoldable_constant dense<[0.0, 0.5, 1.0, 5.0]> : tensor<4xf32>
  %result = "stablehlo.log_plus_one"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[0.0, 0.4054651, 0.6931472, 1.7917595]> : tensor<4xf32>) : tensor<4xf32>
  return
}

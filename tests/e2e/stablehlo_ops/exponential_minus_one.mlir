func.func @exponential_minus_one() {
  %input = util.unfoldable_constant dense<[0.0, 0.5, 1.0, -1.0]> : tensor<4xf32>
  %result = "stablehlo.exponential_minus_one"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[0.0, 0.6487213, 1.7182818, -0.6321205]> : tensor<4xf32>) : tensor<4xf32>
  return
}

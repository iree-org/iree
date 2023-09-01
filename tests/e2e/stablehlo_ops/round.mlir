func.func @tensor() {
  %input = util.unfoldable_constant dense<[-0.7, -0.5, -0.2, 0.0, 0.2, 0.5, 0.7]> : tensor<7xf32>
  %result = "stablehlo.round_nearest_afz"(%input) : (tensor<7xf32>) -> tensor<7xf32>
  check.expect_almost_eq_const(%result, dense<[-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0]> : tensor<7xf32>) : tensor<7xf32>
  return
}


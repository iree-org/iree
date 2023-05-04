func.func @bitcast() {
  %input = util.unfoldable_constant dense<0> : tensor<4xi32>
  %result = "stablehlo.bitcast_convert"(%input) : (tensor<4xi32>) -> tensor<4xf32>
  check.expect_eq_const(%result, dense<0.0> : tensor<4xf32>) : tensor<4xf32>
  return
}

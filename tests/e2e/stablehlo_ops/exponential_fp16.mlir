func.func @tensor_fp16() {
  %input = util.unfoldable_constant dense<[0.0, 1.0, 2.0, 4.0]> : tensor<4xf16>
  %result = stablehlo.exponential %input : (tensor<4xf16>) -> tensor<4xf16>
  check.expect_almost_eq_const(%result, dense<[1.0, 2.7183, 7.3891, 54.5981]> : tensor<4xf16>) : tensor<4xf16>
  return
}

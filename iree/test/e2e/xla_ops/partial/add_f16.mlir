func.func @add_f16() {
  %0 = util.unfoldable_constant dense<[1.5, 2.0, 3.0, 4.2]> : tensor<4xf16>
  %1 = util.unfoldable_constant dense<[5.0, 6.2, 7.0, 8.1]> : tensor<4xf16>
  %result = "mhlo.add"(%0, %1) : (tensor<4xf16>, tensor<4xf16>) -> tensor<4xf16>
  check.expect_almost_eq_const(%result, dense<[6.5, 8.2, 10.0, 12.3]> : tensor<4xf16>) : tensor<4xf16>
  return
}

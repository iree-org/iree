func.func @add_i64() {
  %0 = util.unfoldable_constant dense<4294967296> : tensor<3x4xi64>
  %1 = util.unfoldable_constant dense<2> : tensor<3x4xi64>
  %result = stablehlo.add %0, %1 : tensor<3x4xi64>
  check.expect_eq_const(%result, dense<4294967298> : tensor<3x4xi64>) : tensor<3x4xi64>
  return
}

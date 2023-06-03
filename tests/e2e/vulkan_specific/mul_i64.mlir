func.func @mul_i64() {
  %0 = util.unfoldable_constant dense<4294967296> : tensor<3x4xi64>
  %1 = util.unfoldable_constant dense<3> : tensor<3x4xi64>
  %result = stablehlo.multiply %0, %1 : tensor<3x4xi64>
  check.expect_eq_const(%result, dense<12884901888> : tensor<3x4xi64>) : tensor<3x4xi64>
  return
}

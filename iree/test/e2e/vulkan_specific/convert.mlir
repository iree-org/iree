// TODO(GH-1687 and GH-1869): Remove the test and enable the test in xla_hlo/
func @narrow_int_i32_i8() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi32>
  %res = "xla_hlo.convert"(%input) : (tensor<3xi32>) -> tensor<3xi8>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi8>) : tensor<3xi8>
  return
}

func @widen_int_i8_i32() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[0, 42]> : tensor<2xi8>
  %res = "xla_hlo.convert"(%input) : (tensor<2xi8>) -> tensor<2xi32>
  check.expect_eq_const(%res, dense<[0, 42]> : tensor<2xi32>) : tensor<2xi32>
  return
}

func @narrow_int_i32_i16() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi32>
  %res = "xla_hlo.convert"(%input) : (tensor<3xi32>) -> tensor<3xi16>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi16>) : tensor<3xi16>
  return
}

func @widen_int_i16_i32() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[0, 42]> : tensor<2xi16>
  %res = "xla_hlo.convert"(%input) : (tensor<2xi16>) -> tensor<2xi32>
  check.expect_eq_const(%res, dense<[0, 42]> : tensor<2xi32>) : tensor<2xi32>
  return
}

func @narrow_int_i64_i32() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi64>
  %res = "xla_hlo.convert"(%input) : (tensor<3xi64>) -> tensor<3xi32>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi32>) : tensor<3xi32>
  return
}

func @widen_int_i32_i64() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi32>
  %res = "xla_hlo.convert"(%input) : (tensor<3xi32>) -> tensor<3xi64>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi64>) : tensor<3xi64>
  return
}

func @int_to_float() {
  %input = iree.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi32>
  %res = "xla_hlo.convert"(%input) : (tensor<3xi32>) -> tensor<3xf32>
  check.expect_almost_eq_const(%res, dense<[-42.0, 0.0, 42.0]> : tensor<3xf32>) : tensor<3xf32>
  return
}

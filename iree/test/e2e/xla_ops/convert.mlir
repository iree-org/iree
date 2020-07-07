func @narrow_int_i32_i8() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi32>
  %res = "mhlo.convert"(%input) : (tensor<3xi32>) -> tensor<3xi8>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi8>) : tensor<3xi8>
  return
}

func @widen_int_i8_i32() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi8>
  %res = "mhlo.convert"(%input) : (tensor<3xi8>) -> tensor<3xi32>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi32>) : tensor<3xi32>
  return
}

func @narrow_int_i32_i16() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi32>
  %res = "mhlo.convert"(%input) : (tensor<3xi32>) -> tensor<3xi16>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi16>) : tensor<3xi16>
  return
}

func @widen_int_i16_i32() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi16>
  %res = "mhlo.convert"(%input) : (tensor<3xi16>) -> tensor<3xi32>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi32>) : tensor<3xi32>
  return
}

func @narrow_int_i64_i32() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi64>
  %res = "mhlo.convert"(%input) : (tensor<3xi64>) -> tensor<3xi32>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi32>) : tensor<3xi32>
  return
}

func @widen_int_i32_i64() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi32>
  %res = "mhlo.convert"(%input) : (tensor<3xi32>) -> tensor<3xi64>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi64>) : tensor<3xi64>
  return
}

func @int_to_float() {
  %input = iree.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi32>
  %res = "mhlo.convert"(%input) : (tensor<3xi32>) -> tensor<3xf32>
  check.expect_almost_eq_const(%res, dense<[-42.0, 0.0, 42.0]> : tensor<3xf32>) : tensor<3xf32>
  return
}

// For casting rules, see
// https://www.tensorflow.org/xla/operation_semantics#convertelementtype
func @float_to_int() {
  %input = iree.unfoldable_constant dense<[-10.5, -4.4, 4.4, 10.5]> : tensor<4xf32>
  %res = "mhlo.convert"(%input) : (tensor<4xf32>) -> tensor<4xi32>
  check.expect_eq_const(%res, dense<[-10, -4, 4, 10]> : tensor<4xi32>) : tensor<4xi32>
  return
}

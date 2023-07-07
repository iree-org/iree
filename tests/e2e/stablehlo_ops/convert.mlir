func.func @narrow_int_i32_i8() {
  %input = util.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi32>
  %res = "stablehlo.convert"(%input) : (tensor<3xi32>) -> tensor<3xi8>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi8>) : tensor<3xi8>
  return
}

func.func @widen_int_i8_i32() {
  %input = util.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi8>
  %res = "stablehlo.convert"(%input) : (tensor<3xi8>) -> tensor<3xi32>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi32>) : tensor<3xi32>
  return
}

func.func @narrow_int_i32_i16() {
  %input = util.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi32>
  %res = "stablehlo.convert"(%input) : (tensor<3xi32>) -> tensor<3xi16>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi16>) : tensor<3xi16>
  return
}

func.func @widen_int_i16_i32() {
  %input = util.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi16>
  %res = "stablehlo.convert"(%input) : (tensor<3xi16>) -> tensor<3xi32>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi32>) : tensor<3xi32>
  return
}

func.func @narrow_int_i64_i32() {
  %input = util.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi64>
  %res = "stablehlo.convert"(%input) : (tensor<3xi64>) -> tensor<3xi32>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi32>) : tensor<3xi32>
  return
}

func.func @widen_int_i32_i64() {
  %input = util.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi32>
  %res = "stablehlo.convert"(%input) : (tensor<3xi32>) -> tensor<3xi64>
  check.expect_eq_const(%res, dense<[-42, 0, 42]> : tensor<3xi64>) : tensor<3xi64>
  return
}

func.func @int_to_float() {
  %input = util.unfoldable_constant dense<[-42, 0, 42]> : tensor<3xi32>
  %res = "stablehlo.convert"(%input) : (tensor<3xi32>) -> tensor<3xf32>
  check.expect_almost_eq_const(%res, dense<[-42.0, 0.0, 42.0]> : tensor<3xf32>) : tensor<3xf32>
  return
}

// TODO(#6160): XLA does not specify the rounding behavior, meaning that we
// can't test something like -10.5 as that could be -11 (roundf) or -10 (rint
// with round-to-even mode).
//
// For casting rules, see
// https://www.tensorflow.org/xla/operation_semantics#convertelementtype
// func.func @float_to_int() {
//   %input = util.unfoldable_constant dense<[-10.5, -4.4, 4.4, 10.5]> : tensor<4xf32>
//   %res = "stablehlo.convert"(%input) : (tensor<4xf32>) -> tensor<4xi32>
//   check.expect_eq_const(%res, dense<[-10, -4, 4, 10]> : tensor<4xi32>) : tensor<4xi32>
//   return
// }

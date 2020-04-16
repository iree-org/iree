func @narrow_int() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<42> : tensor<1xi32>
  %0 = "xla_hlo.convert"(%input) : (tensor<1xi32>) -> tensor<1xi8>
  check.expect_eq_const(%0, dense<42> : tensor<1xi8>) : tensor<1xi8>
  return
}

func @widen_int() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<42> : tensor<1xi32>
  %0 = "xla_hlo.convert"(%input) : (tensor<1xi32>) -> tensor<1xi64>
  check.expect_eq_const(%0, dense<42> : tensor<1xi64>) : tensor<1xi64>
  return
}

func @int_to_float() {
  %input = iree.unfoldable_constant dense<42> : tensor<4xi32>
  %0 = "xla_hlo.convert"(%input) : (tensor<4xi32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%0, dense<42.0> : tensor<4xf32>) : tensor<4xf32>
  return
}

// For casting rules, see
// https://www.tensorflow.org/xla/operation_semantics#convertelementtype
func @float_to_int() {
  %input = iree.unfoldable_constant dense<[-10.5, -4.4, 4.4, 10.5]> : tensor<4xf32>
  %0 = "xla_hlo.convert"(%input) : (tensor<4xf32>) -> tensor<4xi32>
  check.expect_eq_const(%0, dense<[-10, -4, 4, 10]> : tensor<4xi32>) : tensor<4xi32>
  return
}

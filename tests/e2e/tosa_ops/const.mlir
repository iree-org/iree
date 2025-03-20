func.func @tensor_float() {
  %result = "tosa.const"() {values = dense<[-1.0, -0.5, 0.0, 1.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[-1.0, -0.5, 0.0, 1.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}

// TODO(#8782): Delete the test.
func.func @matvec() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<250x1024xf32>
  %rhs = util.unfoldable_constant dense<0.5> : tensor<1024xf32>
  %res = "mhlo.dot"(%lhs, %rhs) : (tensor<250x1024xf32>, tensor<1024xf32>) -> tensor<250xf32>
  check.expect_almost_eq_const(%res, dense<512.0> : tensor<250xf32>) : tensor<250xf32>
  return
}

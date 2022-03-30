// large aligned case that can be vectorized and goes through fast path of
// memory promotion and pipelining.
func.func @large_aligned() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<2048x1024xf32>
  %rhs = util.unfoldable_constant dense<0.4> : tensor<1024x512xf32>
  %res = "mhlo.dot"(%lhs, %rhs) : (tensor<2048x1024xf32>, tensor<1024x512xf32>) -> tensor<2048x512xf32>
  check.expect_almost_eq_const(%res, dense<409.596> : tensor<2048x512xf32>) : tensor<2048x512xf32>
  return
}

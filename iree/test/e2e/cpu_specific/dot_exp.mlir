func @dot_exp() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<384x512xf32>
  %rhs = util.unfoldable_constant dense<0.0> : tensor<512x128xf32>
  %0 = "mhlo.dot"(%lhs, %rhs) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
  %1 = "mhlo.exponential"(%0) : (tensor<384x128xf32>) -> tensor<384x128xf32>
  check.expect_almost_eq_const(%1, dense<1.0> : tensor<384x128xf32>) : tensor<384x128xf32>
  return
}

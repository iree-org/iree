func @dot(%lhs: tensor<32x1024xf32>, %rhs: tensor<1024x64xf32>) attributes { iree.module.export } {
  %res = "mhlo.dot"(%lhs, %rhs) : (tensor<32x1024xf32>, tensor<1024x64xf32>) -> tensor<32x64xf32>
  check.expect_almost_eq_const(%res, dense<409.596> : tensor<32x64xf32>) : tensor<32x64xf32>
  return
}

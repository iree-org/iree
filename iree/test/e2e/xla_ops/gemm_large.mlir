func @large_matmul() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<1.0> : tensor<32x1024xf32>
  %rhs = iree.unfoldable_constant dense<0.4> : tensor<1024x64xf32>
  %res = "xla_hlo.dot"(%lhs, %rhs) : (tensor<32x1024xf32>, tensor<1024x64xf32>) -> tensor<32x64xf32>
  check.expect_almost_eq_const(%res, dense<409.596> : tensor<32x64xf32>) : tensor<32x64xf32>
  return
}

func @multiply () attributes { iree.module.export } {
  %c2 = iree.unfoldable_constant dense<2.0> : tensor<f32>
  %res = "mhlo.multiply"(%c2, %c2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%res, dense<4.0> : tensor<f32>) : tensor<f32>
  return
}

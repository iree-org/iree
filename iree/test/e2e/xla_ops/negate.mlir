func @tensor() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[-1.0, -2.0, 3.0, 4.0]> : tensor<4xf32>
  %result = "mhlo.negate"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[1.0, 2.0, -3.0, -4.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func @scalar() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<-4.0> : tensor<f32>
  %result = "mhlo.negate"(%input) : (tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<4.0> : tensor<f32>) : tensor<f32>
  return
}

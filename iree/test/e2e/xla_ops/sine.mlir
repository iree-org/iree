func @tensor() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[0.0, 1.0, 1.5, 2.0]> : tensor<4xf32>
  %result = "mhlo.sine"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[0.0, 0.8415, 0.9975, 0.9093]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func @scalar() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<3.0> : tensor<f32>
  %result = "mhlo.sine"(%input) : (tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<0.14112> : tensor<f32>) : tensor<f32>
  return
}

func @tensor() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[0.0, 1.1, 2.5, 4.9]> : tensor<4xf32>
  %result = "mhlo.floor"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[0.0, 1.0, 2.0, 4.0]> : tensor<4xf32>): tensor<4xf32>
  return
}

func @scalar() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<101.3> : tensor<f32>
  %result = "mhlo.floor"(%input) : (tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<101.0> : tensor<f32>): tensor<f32>
  return
}

func @negative() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<-1.1> : tensor<f32>
  %result = "mhlo.floor"(%input) : (tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<-2.0> : tensor<f32>): tensor<f32>
  return
}

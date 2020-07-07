func @tensor() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %result = "mhlo.log"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[0.0, 0.693147, 1.09861, 1.38629]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func @scalar() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<4.0> : tensor<f32>
  %result = "mhlo.log"(%input) : (tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<1.3863> : tensor<f32>) : tensor<f32>
  return
}

func @double() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<4.0> : tensor<f64>
  %result = "mhlo.log"(%input) : (tensor<f64>) -> tensor<f64>
  check.expect_almost_eq_const(%result, dense<1.3863> : tensor<f64>) : tensor<f64>
  return
}

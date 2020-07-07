func @tensor() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[0.0, 1.0, 2.0, 4.0]> : tensor<4xf32>
  %result = "mhlo.exponential"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[1.0, 2.7183, 7.3891, 54.5981]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func @scalar() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<1.0> : tensor<f32>
  %result = "mhlo.exponential"(%input) : (tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<2.7183> : tensor<f32>) : tensor<f32>
  return
}

func @double() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<1.0> : tensor<f64>
  %result = "mhlo.exponential"(%input) : (tensor<f64>) -> tensor<f64>
  check.expect_almost_eq_const(%result, dense<2.7183> : tensor<f64>) : tensor<f64>
  return
}

func @negative() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<-1.0> : tensor<f32>
  %result = "mhlo.exponential"(%input) : (tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<0.367879> : tensor<f32>) : tensor<f32>
  return
}

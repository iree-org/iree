func @tensor() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<[1.0, 2.0, 7.0, 4.0]> : tensor<4xf32>
  %rhs = iree.unfoldable_constant dense<[5.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %result = "xla_hlo.minimum"(%lhs, %rhs) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func @scalar() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<1.0> : tensor<f32>
  %rhs = iree.unfoldable_constant dense<2.0> : tensor<f32>
  %result = "xla_hlo.minimum"(%lhs, %rhs) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<1.0> : tensor<f32>) : tensor<f32>
  return
}

func @double() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<1.0> : tensor<f64>
  %rhs = iree.unfoldable_constant dense<2.0> : tensor<f64>
  %result = "xla_hlo.minimum"(%lhs, %rhs) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  check.expect_almost_eq_const(%result, dense<1.0> : tensor<f64>) : tensor<f64>
  return
}

func @negative() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<1.0> : tensor<f32>
  %rhs = iree.unfoldable_constant dense<-2.0> : tensor<f32>
  %result = "xla_hlo.minimum"(%lhs, %rhs) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<-2.0> : tensor<f32>) : tensor<f32>
  return
}

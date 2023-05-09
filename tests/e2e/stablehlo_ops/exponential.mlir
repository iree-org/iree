func.func @tensor() {
  %input = util.unfoldable_constant dense<[0.0, 1.0, 2.0, 4.0]> : tensor<4xf32>
  %result = "stablehlo.exponential"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[1.0, 2.7183, 7.3891, 54.5981]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func.func @scalar() {
  %input = util.unfoldable_constant dense<1.0> : tensor<f32>
  %result = "stablehlo.exponential"(%input) : (tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<2.7183> : tensor<f32>) : tensor<f32>
  return
}

func.func @double() {
  %input = util.unfoldable_constant dense<1.0> : tensor<f64>
  %result = "stablehlo.exponential"(%input) : (tensor<f64>) -> tensor<f64>
  check.expect_almost_eq_const(%result, dense<2.7183> : tensor<f64>) : tensor<f64>
  return
}

func.func @negative() {
  %input = util.unfoldable_constant dense<-1.0> : tensor<f32>
  %result = "stablehlo.exponential"(%input) : (tensor<f32>) -> tensor<f32>
  check.expect_almost_eq_const(%result, dense<0.367879> : tensor<f32>) : tensor<f32>
  return
}

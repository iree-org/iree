func.func @rng_normal_2d() {
  %mu = util.unfoldable_constant dense<0.0> : tensor<f32>
  %sigma = util.unfoldable_constant dense<1.0> : tensor<f32>
  %shape = util.unfoldable_constant dense<[3, 5]>  : tensor<2xi64>
  %res = stablehlo.rng %mu, %sigma, %shape, distribution = NORMAL : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<3x5xf32>
  check.expect_almost_eq_const(%res,
    dense<[[0.570861, 0.317593, -0.726538, 1.45925, -1.59632],
           [-0.639956, 0.703875, -0.8801, -0.848389, -0.453391],
           [0.645563, 0.543174, 0.2255, 0.0809385, -1.17198]]> : tensor<3x5xf32>) : tensor<3x5xf32>
  return
}

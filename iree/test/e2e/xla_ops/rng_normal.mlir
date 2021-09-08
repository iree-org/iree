func @rng_normal_2d() {
  %mu = util.unfoldable_constant dense<0.0> : tensor<f32>
  %sigma = util.unfoldable_constant dense<1.0> : tensor<f32>
  %shape = util.unfoldable_constant dense<[3, 5]>  : tensor<2xi64>
  %res = "mhlo.rng_normal"(%mu, %sigma, %shape) : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<3x5xf32>
  check.expect_almost_eq_const(%res,
    dense<[[4.74049,   -0.62986, -1.04619, 0.332649, 2.32022],
           [-0.850417, -0.907853, 0.40775, 1.72218,  1.5235],
           [-0.486889, -0.640311, 1.33677, 1.33779, -0.63425]]> : tensor<3x5xf32>) : tensor<3x5xf32>
  return
}

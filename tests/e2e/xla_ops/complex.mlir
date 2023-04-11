func.func @math_sin() {
  %real = util.unfoldable_constant dense<[0., 1., 1., -1.]> : tensor<4xf32>
  %imag = util.unfoldable_constant dense<[0., 1., -1., 1.]> : tensor<4xf32>
  %complex = "mhlo.complex"(%real, %imag) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xcomplex<f32>>
  %result = "mhlo.sine"(%complex) : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  %result_real = "mhlo.real"(%result) : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  %result_imag = "mhlo.imag"(%result) : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result_real, dense<[0., 1.29846, 1.29846, -1.29846]> : tensor<4xf32>) : tensor<4xf32>
  check.expect_almost_eq_const(%result_imag, dense<[0., 0.634964, -0.634964, 0.634964]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func.func @math_exp() {
  %real = util.unfoldable_constant dense<[0., 1., 1., -1.]> : tensor<4xf32>
  %imag = util.unfoldable_constant dense<[0., 1., -1., 1.]> : tensor<4xf32>
  %complex = "mhlo.complex"(%real, %imag) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xcomplex<f32>>
  %result = "mhlo.exponential"(%complex) : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  %result_real = "mhlo.real"(%result) : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  %result_imag = "mhlo.imag"(%result) : (tensor<4xcomplex<f32>>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result_real, dense<[1., 1.46869, 1.46869, 0.19876]> : tensor<4xf32>) : tensor<4xf32>
  check.expect_almost_eq_const(%result_imag, dense<[0., 2.28735, -2.28735, 0.30956]> : tensor<4xf32>) : tensor<4xf32>
  return
}

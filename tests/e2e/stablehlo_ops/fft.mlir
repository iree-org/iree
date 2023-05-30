// TODO(hanchung): Add other types of fft tests, e.g. fft, ifft, irfft.

func.func @rfft_1d() {
  %input = util.unfoldable_constant dense<[
    9.0, 1.0, 4.5, -0.3, 10.0, -1.0, 5.5, 0.3, 299.0, 3.5, -0.777, 2.0, 1.7,
    3.5, -4.5, 0.0, 9.0, 1.0, 4.5, -0.3, 10.0, -1.0, 5.5, 0.3, 299.0, 3.5,
    -0.777, 2.0, 1.7, 3.5, -4.5, 0.0]> : tensor<32xf32>
  %0 = stablehlo.fft %input, type = RFFT, length = [32] : (tensor<32xf32>) -> tensor<17xcomplex<f32>>
  %1 = stablehlo.real %0 : (tensor<17xcomplex<f32>>) -> tensor<17xf32>
  %2 = stablehlo.imag %0 : (tensor<17xcomplex<f32>>) -> tensor<17xf32>
  check.expect_almost_eq_const(%1, dense<[666.8460, 0.0, -590.16925, 0.0, 593.4485, 0.0, -579.52875, 0.0, 629.95404, 0.0, -567.1126, 0.0, 591.75146, 0.0, -583.1894, 0.0, 630.846]> : tensor<17xf32>) : tensor<17xf32>
  check.expect_almost_eq_const(%2, dense<[0.0, 0.0, -23.956373, 0.0, -10.254326, 0.0, -6.1443653, 0.0, -10.0, 0.0, 3.865515, 0.0, 0.63767385, 0.0, 52.453506, 0.0, 0.0]> : tensor<17xf32>) : tensor<17xf32>
  return
}

func.func @rfft_2d() {
  %input = util.unfoldable_constant dense<[[
    9.0, 1.0, 4.5, -0.3, 10.0, -1.0, 5.5, 0.3, 299.0, 3.5, -0.777, 2.0, 1.7,
    3.5, -4.5, 0.0, 9.0, 1.0, 4.5, -0.3, 10.0, -1.0, 5.5, 0.3, 299.0, 3.5,
    -0.777, 2.0, 1.7, 3.5, -4.5, 0.0]]> : tensor<1x32xf32>
  %0 = stablehlo.fft %input, type = RFFT, length = [32] : (tensor<1x32xf32>) -> tensor<1x17xcomplex<f32>>
  %1 = stablehlo.real %0 : (tensor<1x17xcomplex<f32>>) -> tensor<1x17xf32>
  %2 = stablehlo.imag %0 : (tensor<1x17xcomplex<f32>>) -> tensor<1x17xf32>
  check.expect_almost_eq_const(%1, dense<[[666.8460, 0.0, -590.16925, 0.0, 593.4485, 0.0, -579.52875, 0.0, 629.95404, 0.0, -567.1126, 0.0, 591.75146, 0.0, -583.1894, 0.0, 630.846]]> : tensor<1x17xf32>) : tensor<1x17xf32>
  check.expect_almost_eq_const(%2, dense<[[0.0, 0.0, -23.956373, 0.0, -10.254326, 0.0, -6.1443653, 0.0, -10.0, 0.0, 3.865515, 0.0, 0.63767385, 0.0, 52.453506, 0.0, 0.0]]> : tensor<1x17xf32>) : tensor<1x17xf32>
  return
}

// TODO(hanchung): Add other types of fft tests, e.g. fft, ifft, irfft.

func @rfft_1d() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[
    9.0, 1.0, 4.5, -0.3, 10.0, -1.0, 5.5, 0.3, 299.0, 3.5, -0.777, 2.0, 1.7,
    3.5, -4.5, 0.0, 9.0, 1.0, 4.5, -0.3, 10.0, -1.0, 5.5, 0.3, 299.0, 3.5,
    -0.777, 2.0, 1.7, 3.5, -4.5, 0.0]> : tensor<32xf32>
  %0 = "mhlo.fft"(%input) {
    fft_length = dense<32> : tensor<i64>, fft_type = "RFFT"
  } : (tensor<32xf32>) -> tensor<17xcomplex<f32>>
  %1 = "mhlo.real"(%0) : (tensor<17xcomplex<f32>>) -> tensor<17xf32>
  %2 = "mhlo.imag"(%0) : (tensor<17xcomplex<f32>>) -> tensor<17xf32>
  check.expect_almost_eq_const(%1, dense<[
    666.8460, 0.0000, -590.1693, 0.0000, 593.4485, -0.0001, -579.5288, -0.0004,
    629.9540, 0.0006, -567.1125, -0.0002, 591.7514, 0.0004, -583.1893, -0.0006,
    630.8460]> : tensor<17xf32>) : tensor<17xf32>
  check.expect_almost_eq_const(%2, dense<[
    0.0000, 0.0001, -23.9563, 0.0000, -10.2544, 0.0000, -6.1443, 0.0000,
    -10.0002, 0.0000, 3.8657, 0.0000, 0.6376, -0.0001, 52.4551, 0.0001,
    -0.0003]> : tensor<17xf32>) : tensor<17xf32>
  return
}

func @rfft_2d() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[[
    9.0, 1.0, 4.5, -0.3, 10.0, -1.0, 5.5, 0.3, 299.0, 3.5, -0.777, 2.0, 1.7,
    3.5, -4.5, 0.0, 9.0, 1.0, 4.5, -0.3, 10.0, -1.0, 5.5, 0.3, 299.0, 3.5,
    -0.777, 2.0, 1.7, 3.5, -4.5, 0.0]]> : tensor<1x32xf32>
  %0 = "mhlo.fft"(%input) {
    fft_length = dense<32> : tensor<1xi64>, fft_type = "RFFT"
  } : (tensor<1x32xf32>) -> tensor<1x17xcomplex<f32>>
  %1 = "mhlo.real"(%0) : (tensor<1x17xcomplex<f32>>) -> tensor<1x17xf32>
  %2 = "mhlo.imag"(%0) : (tensor<1x17xcomplex<f32>>) -> tensor<1x17xf32>
  check.expect_almost_eq_const(%1, dense<[
    [666.8460, 0.0000, -590.1693, 0.0000, 593.4485, -0.0001, -579.5288, -0.0004,
     629.9540, 0.0006, -567.1125, -0.0002, 591.7514, 0.0004, -583.1893, -0.0006,
     630.8460]]> : tensor<1x17xf32>) : tensor<1x17xf32>
  check.expect_almost_eq_const(%2, dense<[
    [0.0000, 0.0001, -23.9563, 0.0000, -10.2544, 0.0000, -6.1443, 0.0000,
     -10.0002, 0.0000, 3.8657, 0.0000, 0.6376, -0.0001, 52.4551, 0.0001,
     -0.0003]]> : tensor<1x17xf32>) : tensor<1x17xf32>
  return
}

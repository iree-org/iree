// TODO(hanchung): Add other types of fft tests, e.g. fft, ifft, irfft.

func @rfft_1d() {
  %input = iree.unfoldable_constant dense<[
    9.0, 1.0, 4.5, -0.3, 10.0, -1.0, 5.5, 0.3, 299.0, 3.5, -0.777, 2.0, 1.7,
    3.5, -4.5, 0.0, 9.0, 1.0, 4.5, -0.3, 10.0, -1.0, 5.5, 0.3, 299.0, 3.5,
    -0.777, 2.0, 1.7, 3.5, -4.5, 0.0]> : tensor<32xf32>
  %0 = "mhlo.fft"(%input) {
    fft_length = dense<32> : tensor<i64>, fft_type = "RFFT"
  } : (tensor<32xf32>) -> tensor<17xcomplex<f32>>
  %1 = "mhlo.real"(%0) : (tensor<17xcomplex<f32>>) -> tensor<17xf32>
  %2 = "mhlo.imag"(%0) : (tensor<17xcomplex<f32>>) -> tensor<17xf32>
  check.expect_eq_const(%1, dense<[
    666.846, -534.235, 417.825, -209.118, 53.9493, 97.5035, -153.281, 133.47,
    -54.5364, -54.5372, 133.471, -153.281, 97.5021, 53.9525, -209.12, 417.826,
    -534.235]> : tensor<17xf32>) : tensor<17xf32>
  check.expect_eq_const(%2, dense<[
    0.0, -237.238, 361.777, -443.839, 459.334, -338.492, 207.904, -88.2097,
    15.6604, -15.6609, 88.2116, -207.905, 338.495, -459.335, 443.839, -361.776,
    237.236]> : tensor<17xf32>) : tensor<17xf32>
  return
}

func @rfft_2d() {
  %input = iree.unfoldable_constant dense<[[
    9.0, 1.0, 4.5, -0.3, 10.0, -1.0, 5.5, 0.3, 299.0, 3.5, -0.777, 2.0, 1.7,
    3.5, -4.5, 0.0, 9.0, 1.0, 4.5, -0.3, 10.0, -1.0, 5.5, 0.3, 299.0, 3.5,
    -0.777, 2.0, 1.7, 3.5, -4.5, 0.0]]> : tensor<1x32xf32>
  %0 = "mhlo.fft"(%input) {
    fft_length = dense<32> : tensor<1xi64>, fft_type = "RFFT"
  } : (tensor<1x32xf32>) -> tensor<1x17xcomplex<f32>>
  %1 = "mhlo.real"(%0) : (tensor<1x17xcomplex<f32>>) -> tensor<1x17xf32>
  %2 = "mhlo.imag"(%0) : (tensor<1x17xcomplex<f32>>) -> tensor<1x17xf32>
  check.expect_eq_const(%1, dense<[
    [666.846, -534.235, 417.825, -209.118, 53.9493, 97.5035, -153.281, 133.47,
     -54.5364, -54.5372, 133.471, -153.281, 97.5021, 53.9525, -209.12, 417.826,
     -534.235]]> : tensor<1x17xf32>) : tensor<1x17xf32>
  check.expect_eq_const(%2, dense<[
    [0.0, -237.238, 361.777, -443.839, 459.334, -338.492, 207.904, -88.2097,
     15.6604, -15.6609, 88.2116, -207.905, 338.495, -459.335, 443.839, -361.776,
     237.236]]> : tensor<1x17xf32>) : tensor<1x17xf32>
  return
}

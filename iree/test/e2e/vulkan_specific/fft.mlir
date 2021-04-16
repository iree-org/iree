// TODO(GH-5444): Delete the test and enable fft.mlir test in e2e/xla_ops
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
  check.expect_eq_const(%1, dense<[
    [666.846, -8.10623E-06, -590.169,
  -3.93391E-06, 593.448, -8.53539E-05, -579.529, -0.000412941, 629.954,
  0.000608444, -567.113, -0.000208378, 591.751, 0.000418544, -583.189,
  -0.000618935, 630.846]]> : tensor<1x17xf32>) : tensor<1x17xf32>
  check.expect_eq_const(%2, dense<[
    [0.0, 5.80549E-05, -23.9563, 2.81334E-05, -10.2544, 0.0, -6.14434, 4.05312E-06, -10.0002, -4.76837E-07, 3.86572, 2.57492E-05, 0.637626, -5.19753E-05, 52.4551, 9.13143E-05, -0.000315252]]> : tensor<1x17xf32>) : tensor<1x17xf32>
  return
}

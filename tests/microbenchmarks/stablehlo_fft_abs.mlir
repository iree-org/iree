//===----------------------------------------------------------------------===//
// rfft + abs ops
//===----------------------------------------------------------------------===//

func.func @rfft_abs_6x1024() -> tensor<6x513xf32> {
  %input = util.unfoldable_constant dense<1.0> : tensor<6x1024xf32>
  %0 = "stablehlo.fft"(%input) {
    fft_length = dense<1024> : tensor<1xi64>,
    fft_type = #stablehlo<fft_type RFFT>
  } : (tensor<6x1024xf32>) -> tensor<6x513xcomplex<f32>>
  %1 = "stablehlo.abs"(%0) : (tensor<6x513xcomplex<f32>>) -> tensor<6x513xf32>
  return %1: tensor<6x513xf32>
}

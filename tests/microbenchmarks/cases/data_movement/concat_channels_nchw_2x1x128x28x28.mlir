// tensor.concat of two NCHW activations along channels, 2 x 1x128x28x28 ->
// 1x256x28x28 f32.

func.func @concat_channels_nchw_2x1x128x28x28(%a: tensor<1x128x28x28xf32>, %b: tensor<1x128x28x28xf32>) -> tensor<1x256x28x28xf32> {
  %r = tensor.concat dim(1) %a, %b : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x256x28x28xf32>
  return %r : tensor<1x256x28x28xf32>
}

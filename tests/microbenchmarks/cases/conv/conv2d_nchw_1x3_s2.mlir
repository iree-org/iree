// Conv 1x3 stride 2 (unit filter extent on a strided axis), NCHW,
// 128 -> 256 channels, 28x29 -> 14x14.

util.global private @weight = #util.byte_pattern<63> : tensor<256x128x1x3xf32>

func.func @conv2d_nchw_1x3_s2(%x: tensor<1x128x28x29xf32>) -> tensor<1x256x14x14xf32> {
  %weight = util.global.load @weight : tensor<256x128x1x3xf32>
  %e1 = tensor.empty() : tensor<1x256x14x14xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
  %conv4 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%x, %weight : tensor<1x128x28x29xf32>, tensor<256x128x1x3xf32>) outs(%f3 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
  return %conv4 : tensor<1x256x14x14xf32>
}

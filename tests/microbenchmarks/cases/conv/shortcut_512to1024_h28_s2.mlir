// ResNet-50 v1.5 downsample shortcut, NCHW: conv 1x1 stride 2, 512 -> 1024
// channels, 28x28 -> 14x14. Plain conv, no producer, no bias.

util.global private @weight = #util.byte_pattern<63> : tensor<1024x512x1x1xf32>

func.func @shortcut_512to1024_h28_s2(%x: tensor<1x512x28x28xf32>) -> tensor<1x1024x14x14xf32> {
  %weight = util.global.load @weight : tensor<1024x512x1x1xf32>
  %e1 = tensor.empty() : tensor<1x1024x14x14xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
  %conv4 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%x, %weight : tensor<1x512x28x28xf32>, tensor<1024x512x1x1xf32>) outs(%f3 : tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
  return %conv4 : tensor<1x1024x14x14xf32>
}

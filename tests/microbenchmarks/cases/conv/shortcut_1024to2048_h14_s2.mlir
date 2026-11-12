// ResNet-50 v1.5 downsample shortcut, NCHW: conv 1x1 stride 2, 1024 -> 2048
// channels, 14x14 -> 7x7. Plain conv, no producer, no bias.

util.global private @weight = #util.byte_pattern<63> : tensor<2048x1024x1x1xf32>

func.func @shortcut_1024to2048_h14_s2(%x: tensor<1x1024x14x14xf32>) -> tensor<1x2048x7x7xf32> {
  %weight = util.global.load @weight : tensor<2048x1024x1x1xf32>
  %e1 = tensor.empty() : tensor<1x2048x7x7xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
  %conv4 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%x, %weight : tensor<1x1024x14x14xf32>, tensor<2048x1024x1x1xf32>) outs(%f3 : tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
  return %conv4 : tensor<1x2048x7x7xf32>
}

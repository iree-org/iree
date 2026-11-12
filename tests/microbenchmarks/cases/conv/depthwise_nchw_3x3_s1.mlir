// Depthwise conv 3x3 stride 1, NCHW/CHW, 32 channels at 112x112.

util.global private @weight = #util.byte_pattern<63> : tensor<32x3x3xf32>

func.func @depthwise_nchw_3x3_s1(%x: tensor<1x32x114x114xf32>) -> tensor<1x32x112x112xf32> {
  %weight = util.global.load @weight : tensor<32x3x3xf32>
  %e1 = tensor.empty() : tensor<1x32x112x112xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
  %conv4 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%x, %weight : tensor<1x32x114x114xf32>, tensor<32x3x3xf32>) outs(%f3 : tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
  return %conv4 : tensor<1x32x112x112xf32>
}

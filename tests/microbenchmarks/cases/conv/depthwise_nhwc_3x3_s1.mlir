// Depthwise conv 3x3 stride 1, NHWC/HWC, 32 channels at 112x112 (MobileNet).

util.global private @weight = #util.byte_pattern<63> : tensor<3x3x32xf32>

func.func @depthwise_nhwc_3x3_s1(%x: tensor<1x114x114x32xf32>) -> tensor<1x112x112x32xf32> {
  %weight = util.global.load @weight : tensor<3x3x32xf32>
  %e1 = tensor.empty() : tensor<1x112x112x32xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  %conv4 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%x, %weight : tensor<1x114x114x32xf32>, tensor<3x3x32xf32>) outs(%f3 : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  return %conv4 : tensor<1x112x112x32xf32>
}

// Depthwise conv 3x3 stride 2, NHWC/HWC, 64 channels, 113x113 -> 56x56.

util.global private @weight = #util.byte_pattern<63> : tensor<3x3x64xf32>

func.func @depthwise_nhwc_3x3_s2(%x: tensor<1x113x113x64xf32>) -> tensor<1x56x56x64xf32> {
  %weight = util.global.load @weight : tensor<3x3x64xf32>
  %e1 = tensor.empty() : tensor<1x56x56x64xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %conv4 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%x, %weight : tensor<1x113x113x64xf32>, tensor<3x3x64xf32>) outs(%f3 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  return %conv4 : tensor<1x56x56x64xf32>
}

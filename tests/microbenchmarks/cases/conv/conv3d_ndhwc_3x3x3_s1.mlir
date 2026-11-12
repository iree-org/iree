// 3-D conv 3x3x3 stride 1, NDHWC/DHWCF, 32 -> 64 channels, 16x28x28 output.

util.global private @weight = #util.byte_pattern<63> : tensor<3x3x3x32x64xf32>

func.func @conv3d_ndhwc_3x3x3_s1(%x: tensor<1x18x30x30x32xf32>) -> tensor<1x16x28x28x64xf32> {
  %weight = util.global.load @weight : tensor<3x3x3x32x64xf32>
  %e1 = tensor.empty() : tensor<1x16x28x28x64xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<1x16x28x28x64xf32>) -> tensor<1x16x28x28x64xf32>
  %conv4 = linalg.conv_3d_ndhwc_dhwcf {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
      ins(%x, %weight : tensor<1x18x30x30x32xf32>, tensor<3x3x3x32x64xf32>) outs(%f3 : tensor<1x16x28x28x64xf32>) -> tensor<1x16x28x28x64xf32>
  return %conv4 : tensor<1x16x28x28x64xf32>
}

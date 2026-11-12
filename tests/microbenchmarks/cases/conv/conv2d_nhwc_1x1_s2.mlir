// Conv 1x1 stride 2, NHWC/HWCF, 512 -> 1024 channels, 28x28 -> 14x14.

util.global private @weight = #util.byte_pattern<63> : tensor<1x1x512x1024xf32>

func.func @conv2d_nhwc_1x1_s2(%x: tensor<1x28x28x512xf32>) -> tensor<1x14x14x1024xf32> {
  %weight = util.global.load @weight : tensor<1x1x512x1024xf32>
  %e1 = tensor.empty() : tensor<1x14x14x1024xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
  %conv4 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%x, %weight : tensor<1x28x28x512xf32>, tensor<1x1x512x1024xf32>) outs(%f3 : tensor<1x14x14x1024xf32>) -> tensor<1x14x14x1024xf32>
  return %conv4 : tensor<1x14x14x1024xf32>
}

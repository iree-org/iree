// Conv 1x1 stride 1, NCHW, 256 -> 64 channels at 56x56 (ResNet bottleneck reduce).

util.global private @weight = #util.byte_pattern<63> : tensor<64x256x1x1xf32>

func.func @conv2d_nchw_1x1_s1(%x: tensor<1x256x56x56xf32>) -> tensor<1x64x56x56xf32> {
  %weight = util.global.load @weight : tensor<64x256x1x1xf32>
  %e1 = tensor.empty() : tensor<1x64x56x56xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %conv4 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%x, %weight : tensor<1x256x56x56xf32>, tensor<64x256x1x1xf32>) outs(%f3 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  return %conv4 : tensor<1x64x56x56xf32>
}

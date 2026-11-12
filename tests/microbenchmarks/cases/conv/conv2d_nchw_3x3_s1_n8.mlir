// Conv 3x3 stride 1, NCHW, batch 8, 64 channels at 28x28 (pre-padded input).

util.global private @weight = #util.byte_pattern<63> : tensor<64x64x3x3xf32>

func.func @conv2d_nchw_3x3_s1_n8(%x: tensor<8x64x30x30xf32>) -> tensor<8x64x28x28xf32> {
  %weight = util.global.load @weight : tensor<64x64x3x3xf32>
  %e1 = tensor.empty() : tensor<8x64x28x28xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<8x64x28x28xf32>) -> tensor<8x64x28x28xf32>
  %conv4 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%x, %weight : tensor<8x64x30x30xf32>, tensor<64x64x3x3xf32>) outs(%f3 : tensor<8x64x28x28xf32>) -> tensor<8x64x28x28xf32>
  return %conv4 : tensor<8x64x28x28xf32>
}

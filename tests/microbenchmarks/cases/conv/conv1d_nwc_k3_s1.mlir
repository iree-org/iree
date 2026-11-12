// 1-D conv, kernel 3 stride 1, NWC/WCF, 256 channels, length 1026 -> 1024.

util.global private @weight = #util.byte_pattern<63> : tensor<3x256x256xf32>

func.func @conv1d_nwc_k3_s1(%x: tensor<1x1026x256xf32>) -> tensor<1x1024x256xf32> {
  %weight = util.global.load @weight : tensor<3x256x256xf32>
  %e1 = tensor.empty() : tensor<1x1024x256xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<1x1024x256xf32>) -> tensor<1x1024x256xf32>
  %conv4 = linalg.conv_1d_nwc_wcf {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
      ins(%x, %weight : tensor<1x1026x256xf32>, tensor<3x256x256xf32>) outs(%f3 : tensor<1x1024x256xf32>) -> tensor<1x1024x256xf32>
  return %conv4 : tensor<1x1024x256xf32>
}

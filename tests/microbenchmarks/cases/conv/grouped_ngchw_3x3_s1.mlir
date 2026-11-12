// Grouped conv 3x3 stride 1, NGCHW/GFCHW, 32 groups x 4 channels at 56x56
// (ResNeXt-50 32x4d, first stage). Pre-padded input.

util.global private @weight = #util.byte_pattern<63> : tensor<32x4x4x3x3xf32>

func.func @grouped_ngchw_3x3_s1(%x: tensor<1x32x4x58x58xf32>) -> tensor<1x32x4x56x56xf32> {
  %weight = util.global.load @weight : tensor<32x4x4x3x3xf32>
  %e1 = tensor.empty() : tensor<1x32x4x56x56xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<1x32x4x56x56xf32>) -> tensor<1x32x4x56x56xf32>
  %conv4 = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%x, %weight : tensor<1x32x4x58x58xf32>, tensor<32x4x4x3x3xf32>) outs(%f3 : tensor<1x32x4x56x56xf32>) -> tensor<1x32x4x56x56xf32>
  return %conv4 : tensor<1x32x4x56x56xf32>
}

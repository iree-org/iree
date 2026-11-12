// Grouped conv 3x3 stride 1, NHWGC/GFHWC, 32 groups x 4 channels at 56x56.
// Pre-padded input.

util.global private @weight = #util.byte_pattern<63> : tensor<32x4x3x3x4xf32>

func.func @grouped_nhwgc_3x3_s1(%x: tensor<1x58x58x32x4xf32>) -> tensor<1x56x56x32x4xf32> {
  %weight = util.global.load @weight : tensor<32x4x3x3x4xf32>
  %e1 = tensor.empty() : tensor<1x56x56x32x4xf32>
  %c2 = arith.constant 0.0 : f32
  %f3 = linalg.fill ins(%c2 : f32) outs(%e1 : tensor<1x56x56x32x4xf32>) -> tensor<1x56x56x32x4xf32>
  %conv4 = linalg.conv_2d_nhwgc_gfhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%x, %weight : tensor<1x58x58x32x4xf32>, tensor<32x4x3x3x4xf32>) outs(%f3 : tensor<1x56x56x32x4xf32>) -> tensor<1x56x56x32x4xf32>
  return %conv4 : tensor<1x56x56x32x4xf32>
}

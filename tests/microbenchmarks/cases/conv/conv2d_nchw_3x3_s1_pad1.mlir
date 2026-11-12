// Conv 3x3 stride 1 with explicit tensor.pad, NCHW, 64 channels at 56x56.

util.global private @weight = #util.byte_pattern<63> : tensor<64x64x3x3xf32>

func.func @conv2d_nchw_3x3_s1_pad1(%x: tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32> {
  %c1 = arith.constant 0.0 : f32
  %p2 = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
      tensor.yield %c1 : f32
  } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
  %weight = util.global.load @weight : tensor<64x64x3x3xf32>
  %e3 = tensor.empty() : tensor<1x64x56x56xf32>
  %c4 = arith.constant 0.0 : f32
  %f5 = linalg.fill ins(%c4 : f32) outs(%e3 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %conv6 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%p2, %weight : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%f5 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  return %conv6 : tensor<1x64x56x56xf32>
}

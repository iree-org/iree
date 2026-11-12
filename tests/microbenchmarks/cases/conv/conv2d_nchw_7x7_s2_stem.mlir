// ResNet stem: tensor.pad 3, conv 7x7 stride 2, NCHW, 3 -> 64 channels,
// 224x224 -> 112x112.

util.global private @weight = #util.byte_pattern<63> : tensor<64x3x7x7xf32>

func.func @conv2d_nchw_7x7_s2_stem(%x: tensor<1x3x224x224xf32>) -> tensor<1x64x112x112xf32> {
  %c1 = arith.constant 0.0 : f32
  %p2 = tensor.pad %x low[0, 0, 3, 3] high[0, 0, 3, 3] {
    ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
      tensor.yield %c1 : f32
  } : tensor<1x3x224x224xf32> to tensor<1x3x230x230xf32>
  %weight = util.global.load @weight : tensor<64x3x7x7xf32>
  %e3 = tensor.empty() : tensor<1x64x112x112xf32>
  %c4 = arith.constant 0.0 : f32
  %f5 = linalg.fill ins(%c4 : f32) outs(%e3 : tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
  %conv6 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%p2, %weight : tensor<1x3x230x230xf32>, tensor<64x3x7x7xf32>) outs(%f5 : tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
  return %conv6 : tensor<1x64x112x112xf32>
}

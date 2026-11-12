// ResNet stem max pool: tensor.pad with -inf, 3x3 window stride 2, NCHW,
// 1x64x112x112 -> 1x64x56x56 f32.

func.func @maxpool_nchw_3x3_s2_pad1(%x: tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32> {
  %ninf = arith.constant 0xFF800000 : f32
  %p = tensor.pad %x low[0, 0, 1, 1] high[0, 0, 1, 1] {
  ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
    tensor.yield %ninf : f32
  } : tensor<1x64x112x112xf32> to tensor<1x64x114x114xf32>
  %win = tensor.empty() : tensor<3x3xf32>
  %e = tensor.empty() : tensor<1x64x56x56xf32>
  %f = linalg.fill ins(%ninf : f32) outs(%e : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %r = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%p, %win : tensor<1x64x114x114xf32>, tensor<3x3xf32>) outs(%f : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  return %r : tensor<1x64x56x56xf32>
}

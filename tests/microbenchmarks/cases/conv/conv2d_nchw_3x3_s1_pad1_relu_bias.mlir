// Relu producer, tensor.pad, conv 3x3 stride 1 with bias, NCHW, 64 channels at 56x56.

util.global private @weight = #util.byte_pattern<63> : tensor<64x64x3x3xf32>
util.global private @bias = #util.byte_pattern<63> : tensor<64xf32>

func.func @conv2d_nchw_3x3_s1_pad1_relu_bias(%x: tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32> {
  %e1 = tensor.empty() : tensor<1x64x56x56xf32>
  %g2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%x : tensor<1x64x56x56xf32>) outs(%e1 : tensor<1x64x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %zero = arith.constant 0.0 : f32
      %r = arith.maximumf %in, %zero : f32
      linalg.yield %r : f32
  } -> tensor<1x64x56x56xf32>
  %c3 = arith.constant 0.0 : f32
  %p4 = tensor.pad %g2 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
      tensor.yield %c3 : f32
  } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
  %weight = util.global.load @weight : tensor<64x64x3x3xf32>
  %bias = util.global.load @bias : tensor<64xf32>
  %e5 = tensor.empty() : tensor<1x64x56x56xf32>
  %b6 = linalg.broadcast ins(%bias : tensor<64xf32>) outs(%e5 : tensor<1x64x56x56xf32>) dimensions = [0, 2, 3]
  %conv7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%p4, %weight : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%b6 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  return %conv7 : tensor<1x64x56x56xf32>
}

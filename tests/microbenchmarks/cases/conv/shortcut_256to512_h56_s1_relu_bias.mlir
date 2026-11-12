// Stride-1 control for shortcut_256to512_h56_s2_relu_bias: same channels and input, conv 1x1
// stride 1, 56x56 -> 56x56, relu producer, bias.

util.global private @weight = #util.byte_pattern<63> : tensor<512x256x1x1xf32>
util.global private @bias = #util.byte_pattern<63> : tensor<512xf32>

func.func @shortcut_256to512_h56_s1_relu_bias(%x: tensor<1x256x56x56xf32>) -> tensor<1x512x56x56xf32> {
  %e1 = tensor.empty() : tensor<1x256x56x56xf32>
  %g2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%x : tensor<1x256x56x56xf32>) outs(%e1 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %zero = arith.constant 0.0 : f32
      %r = arith.maximumf %in, %zero : f32
      linalg.yield %r : f32
  } -> tensor<1x256x56x56xf32>
  %weight = util.global.load @weight : tensor<512x256x1x1xf32>
  %bias = util.global.load @bias : tensor<512xf32>
  %e3 = tensor.empty() : tensor<1x512x56x56xf32>
  %b4 = linalg.broadcast ins(%bias : tensor<512xf32>) outs(%e3 : tensor<1x512x56x56xf32>) dimensions = [0, 2, 3]
  %conv5 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%g2, %weight : tensor<1x256x56x56xf32>, tensor<512x256x1x1xf32>) outs(%b4 : tensor<1x512x56x56xf32>) -> tensor<1x512x56x56xf32>
  return %conv5 : tensor<1x512x56x56xf32>
}

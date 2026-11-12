// QDQ variant of conv/shortcut_256to512_h56_s2_relu_bias: relu, per-tensor activation
// quantization (zero point -128), i8 weights with per-channel scale, bias,
// requantized i8 output.

util.global private @weight = #util.byte_pattern<63> : tensor<512x256x1x1xi8>
util.global private @weight_scale = #util.byte_pattern<63> : tensor<512xf32>
util.global private @bias = #util.byte_pattern<63> : tensor<512xf32>

func.func @shortcut_256to512_h56_s2_relu_bias_qdq(%x: tensor<1x256x56x56xf32>) -> tensor<1x512x28x28xi8> {
  %e1 = tensor.empty() : tensor<1x256x56x56xf32>
  %g2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%x : tensor<1x256x56x56xf32>) outs(%e1 : tensor<1x256x56x56xf32>) {
    ^bb0(%in: f32, %out: f32):
      %zero = arith.constant 0.0 : f32
      %r = arith.maximumf %in, %zero : f32
      linalg.yield %r : f32
  } -> tensor<1x256x56x56xf32>
  %c3 = arith.constant 0.05 : f32
  %c4 = arith.constant -128 : i8
  %e5 = tensor.empty() : tensor<1x256x56x56xi8>
  %q6 = iree_linalg_ext.quantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], quant_min = -128, quant_max = 127>
      ins(%g2, %c3, %c4 : tensor<1x256x56x56xf32>, f32, i8) outs(%e5 : tensor<1x256x56x56xi8>) -> tensor<1x256x56x56xi8>
  %e7 = tensor.empty() : tensor<1x256x56x56xf32>
  %dq8 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]>
      ins(%q6, %c3, %c4 : tensor<1x256x56x56xi8>, f32, i8) outs(%e7 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
  %weight = util.global.load @weight : tensor<512x256x1x1xi8>
  %weight_scale = util.global.load @weight_scale : tensor<512xf32>
  %c9 = arith.constant dense<0> : tensor<512xi8>
  %e10 = tensor.empty() : tensor<512x256x1x1xf32>
  %wdq11 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]>
      ins(%weight, %weight_scale, %c9 : tensor<512x256x1x1xi8>, tensor<512xf32>, tensor<512xi8>) outs(%e10 : tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf32>
  %bias = util.global.load @bias : tensor<512xf32>
  %e12 = tensor.empty() : tensor<1x512x28x28xf32>
  %b13 = linalg.broadcast ins(%bias : tensor<512xf32>) outs(%e12 : tensor<1x512x28x28xf32>) dimensions = [0, 2, 3]
  %conv14 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%dq8, %wdq11 : tensor<1x256x56x56xf32>, tensor<512x256x1x1xf32>) outs(%b13 : tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
  %c15 = arith.constant 0.1 : f32
  %c16 = arith.constant -128 : i8
  %e17 = tensor.empty() : tensor<1x512x28x28xi8>
  %q18 = iree_linalg_ext.quantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], quant_min = -128, quant_max = 127>
      ins(%conv14, %c15, %c16 : tensor<1x512x28x28xf32>, f32, i8) outs(%e17 : tensor<1x512x28x28xi8>) -> tensor<1x512x28x28xi8>
  return %q18 : tensor<1x512x28x28xi8>
}

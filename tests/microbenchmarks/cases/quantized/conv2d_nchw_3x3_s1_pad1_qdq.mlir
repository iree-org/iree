// QDQ conv 3x3 stride 1 with tensor.pad, NCHW, 64 channels at 56x56, bias.
// Per-tensor activation quantization (zero point -128), i8 weights with per-channel
// scale, requantized i8 output.

util.global private @weight = #util.byte_pattern<63> : tensor<64x64x3x3xi8>
util.global private @weight_scale = #util.byte_pattern<63> : tensor<64xf32>
util.global private @bias = #util.byte_pattern<63> : tensor<64xf32>

func.func @conv2d_nchw_3x3_s1_pad1_qdq(%x: tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xi8> {
  %c1 = arith.constant 0.05 : f32
  %c2 = arith.constant -128 : i8
  %e3 = tensor.empty() : tensor<1x64x56x56xi8>
  %q4 = iree_linalg_ext.quantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], quant_min = -128, quant_max = 127>
      ins(%x, %c1, %c2 : tensor<1x64x56x56xf32>, f32, i8) outs(%e3 : tensor<1x64x56x56xi8>) -> tensor<1x64x56x56xi8>
  %e5 = tensor.empty() : tensor<1x64x56x56xf32>
  %dq6 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]>
      ins(%q4, %c1, %c2 : tensor<1x64x56x56xi8>, f32, i8) outs(%e5 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %c7 = arith.constant 0.0 : f32
  %p8 = tensor.pad %dq6 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
      tensor.yield %c7 : f32
  } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
  %weight = util.global.load @weight : tensor<64x64x3x3xi8>
  %weight_scale = util.global.load @weight_scale : tensor<64xf32>
  %c9 = arith.constant dense<0> : tensor<64xi8>
  %e10 = tensor.empty() : tensor<64x64x3x3xf32>
  %wdq11 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]>
      ins(%weight, %weight_scale, %c9 : tensor<64x64x3x3xi8>, tensor<64xf32>, tensor<64xi8>) outs(%e10 : tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
  %bias = util.global.load @bias : tensor<64xf32>
  %e12 = tensor.empty() : tensor<1x64x56x56xf32>
  %b13 = linalg.broadcast ins(%bias : tensor<64xf32>) outs(%e12 : tensor<1x64x56x56xf32>) dimensions = [0, 2, 3]
  %conv14 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%p8, %wdq11 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%b13 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %c15 = arith.constant 0.1 : f32
  %c16 = arith.constant -128 : i8
  %e17 = tensor.empty() : tensor<1x64x56x56xi8>
  %q18 = iree_linalg_ext.quantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], quant_min = -128, quant_max = 127>
      ins(%conv14, %c15, %c16 : tensor<1x64x56x56xf32>, f32, i8) outs(%e17 : tensor<1x64x56x56xi8>) -> tensor<1x64x56x56xi8>
  return %q18 : tensor<1x64x56x56xi8>
}

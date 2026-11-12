// QDQ conv 3x3 stride 2 with tensor.pad, NCHW, 128 channels, 56x56 -> 28x28.
// Per-tensor activation quantization (zero point -128), i8 weights with per-channel
// scale, requantized i8 output.

util.global private @weight = #util.byte_pattern<63> : tensor<128x128x3x3xi8>
util.global private @weight_scale = #util.byte_pattern<63> : tensor<128xf32>

func.func @conv2d_nchw_3x3_s2_pad1_qdq(%x: tensor<1x128x56x56xf32>) -> tensor<1x128x28x28xi8> {
  %c1 = arith.constant 0.05 : f32
  %c2 = arith.constant -128 : i8
  %e3 = tensor.empty() : tensor<1x128x56x56xi8>
  %q4 = iree_linalg_ext.quantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], quant_min = -128, quant_max = 127>
      ins(%x, %c1, %c2 : tensor<1x128x56x56xf32>, f32, i8) outs(%e3 : tensor<1x128x56x56xi8>) -> tensor<1x128x56x56xi8>
  %e5 = tensor.empty() : tensor<1x128x56x56xf32>
  %dq6 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]>
      ins(%q4, %c1, %c2 : tensor<1x128x56x56xi8>, f32, i8) outs(%e5 : tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
  %c7 = arith.constant 0.0 : f32
  %p8 = tensor.pad %dq6 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
      tensor.yield %c7 : f32
  } : tensor<1x128x56x56xf32> to tensor<1x128x58x58xf32>
  %weight = util.global.load @weight : tensor<128x128x3x3xi8>
  %weight_scale = util.global.load @weight_scale : tensor<128xf32>
  %c9 = arith.constant dense<0> : tensor<128xi8>
  %e10 = tensor.empty() : tensor<128x128x3x3xf32>
  %wdq11 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]>
      ins(%weight, %weight_scale, %c9 : tensor<128x128x3x3xi8>, tensor<128xf32>, tensor<128xi8>) outs(%e10 : tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
  %e12 = tensor.empty() : tensor<1x128x28x28xf32>
  %c13 = arith.constant 0.0 : f32
  %f14 = linalg.fill ins(%c13 : f32) outs(%e12 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
  %conv15 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%p8, %wdq11 : tensor<1x128x58x58xf32>, tensor<128x128x3x3xf32>) outs(%f14 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
  %c16 = arith.constant 0.1 : f32
  %c17 = arith.constant -128 : i8
  %e18 = tensor.empty() : tensor<1x128x28x28xi8>
  %q19 = iree_linalg_ext.quantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], quant_min = -128, quant_max = 127>
      ins(%conv15, %c16, %c17 : tensor<1x128x28x28xf32>, f32, i8) outs(%e18 : tensor<1x128x28x28xi8>) -> tensor<1x128x28x28xi8>
  return %q19 : tensor<1x128x28x28xi8>
}

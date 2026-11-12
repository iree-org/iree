// QDQ conv 1x3 stride 2, NCHW, 128 -> 256 channels, 28x29 -> 14x14.
// Per-tensor activation quantization (zero point -128), i8 weights with per-channel
// scale, requantized i8 output.

util.global private @weight = #util.byte_pattern<63> : tensor<256x128x1x3xi8>
util.global private @weight_scale = #util.byte_pattern<63> : tensor<256xf32>

func.func @conv2d_nchw_1x3_s2_qdq(%x: tensor<1x128x28x29xf32>) -> tensor<1x256x14x14xi8> {
  %c1 = arith.constant 0.05 : f32
  %c2 = arith.constant -128 : i8
  %e3 = tensor.empty() : tensor<1x128x28x29xi8>
  %q4 = iree_linalg_ext.quantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], quant_min = -128, quant_max = 127>
      ins(%x, %c1, %c2 : tensor<1x128x28x29xf32>, f32, i8) outs(%e3 : tensor<1x128x28x29xi8>) -> tensor<1x128x28x29xi8>
  %e5 = tensor.empty() : tensor<1x128x28x29xf32>
  %dq6 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]>
      ins(%q4, %c1, %c2 : tensor<1x128x28x29xi8>, f32, i8) outs(%e5 : tensor<1x128x28x29xf32>) -> tensor<1x128x28x29xf32>
  %weight = util.global.load @weight : tensor<256x128x1x3xi8>
  %weight_scale = util.global.load @weight_scale : tensor<256xf32>
  %c7 = arith.constant dense<0> : tensor<256xi8>
  %e8 = tensor.empty() : tensor<256x128x1x3xf32>
  %wdq9 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]>
      ins(%weight, %weight_scale, %c7 : tensor<256x128x1x3xi8>, tensor<256xf32>, tensor<256xi8>) outs(%e8 : tensor<256x128x1x3xf32>) -> tensor<256x128x1x3xf32>
  %e10 = tensor.empty() : tensor<1x256x14x14xf32>
  %c11 = arith.constant 0.0 : f32
  %f12 = linalg.fill ins(%c11 : f32) outs(%e10 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
  %conv13 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%dq6, %wdq9 : tensor<1x128x28x29xf32>, tensor<256x128x1x3xf32>) outs(%f12 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
  %c14 = arith.constant 0.1 : f32
  %c15 = arith.constant -128 : i8
  %e16 = tensor.empty() : tensor<1x256x14x14xi8>
  %q17 = iree_linalg_ext.quantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], quant_min = -128, quant_max = 127>
      ins(%conv13, %c14, %c15 : tensor<1x256x14x14xf32>, f32, i8) outs(%e16 : tensor<1x256x14x14xi8>) -> tensor<1x256x14x14xi8>
  return %q17 : tensor<1x256x14x14xi8>
}

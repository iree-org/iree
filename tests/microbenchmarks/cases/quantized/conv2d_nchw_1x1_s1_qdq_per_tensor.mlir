// QDQ conv 1x1 stride 1, NCHW, 256 -> 64 channels at 56x56.
// Per-tensor activation quantization (zero point -128), i8 weights with per-tensor
// scale, requantized i8 output.

util.global private @weight = #util.byte_pattern<63> : tensor<64x256x1x1xi8>

func.func @conv2d_nchw_1x1_s1_qdq_per_tensor(%x: tensor<1x256x56x56xf32>) -> tensor<1x64x56x56xi8> {
  %c1 = arith.constant 0.05 : f32
  %c2 = arith.constant -128 : i8
  %e3 = tensor.empty() : tensor<1x256x56x56xi8>
  %q4 = iree_linalg_ext.quantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], quant_min = -128, quant_max = 127>
      ins(%x, %c1, %c2 : tensor<1x256x56x56xf32>, f32, i8) outs(%e3 : tensor<1x256x56x56xi8>) -> tensor<1x256x56x56xi8>
  %e5 = tensor.empty() : tensor<1x256x56x56xf32>
  %dq6 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]>
      ins(%q4, %c1, %c2 : tensor<1x256x56x56xi8>, f32, i8) outs(%e5 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
  %weight = util.global.load @weight : tensor<64x256x1x1xi8>
  %c7 = arith.constant 0.02 : f32
  %c8 = arith.constant 0 : i8
  %e9 = tensor.empty() : tensor<64x256x1x1xf32>
  %wdq10 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]>
      ins(%weight, %c7, %c8 : tensor<64x256x1x1xi8>, f32, i8) outs(%e9 : tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xf32>
  %e11 = tensor.empty() : tensor<1x64x56x56xf32>
  %c12 = arith.constant 0.0 : f32
  %f13 = linalg.fill ins(%c12 : f32) outs(%e11 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %conv14 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%dq6, %wdq10 : tensor<1x256x56x56xf32>, tensor<64x256x1x1xf32>) outs(%f13 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %c15 = arith.constant 0.1 : f32
  %c16 = arith.constant -128 : i8
  %e17 = tensor.empty() : tensor<1x64x56x56xi8>
  %q18 = iree_linalg_ext.quantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], quant_min = -128, quant_max = 127>
      ins(%conv14, %c15, %c16 : tensor<1x64x56x56xf32>, f32, i8) outs(%e17 : tensor<1x64x56x56xi8>) -> tensor<1x64x56x56xi8>
  return %q18 : tensor<1x64x56x56xi8>
}

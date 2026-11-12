// Standalone activation quantize + dequantize (per-tensor, zero point -128),
// 1x256x56x56 f32 -> i8 -> f32.

func.func @quantize_dequantize_1x256x56x56(%x: tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32> {
  %c1 = arith.constant 0.05 : f32
  %c2 = arith.constant -128 : i8
  %e3 = tensor.empty() : tensor<1x256x56x56xi8>
  %q4 = iree_linalg_ext.quantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], quant_min = -128, quant_max = 127>
      ins(%x, %c1, %c2 : tensor<1x256x56x56xf32>, f32, i8) outs(%e3 : tensor<1x256x56x56xi8>) -> tensor<1x256x56x56xi8>
  %e5 = tensor.empty() : tensor<1x256x56x56xf32>
  %dq6 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]>
      ins(%q4, %c1, %c2 : tensor<1x256x56x56xi8>, f32, i8) outs(%e5 : tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
  return %dq6 : tensor<1x256x56x56xf32>
}

// QDQ batch matmul of two activations, BxMxNxK = 12x384x384x64. Both operands
// per-tensor quantized with zero point -128, f32 output.

func.func @batch_matmul_12x384x384x64_qdq(%a: tensor<12x384x64xf32>, %b: tensor<12x64x384xf32>) -> tensor<12x384x384xf32> {
  %c1 = arith.constant 0.05 : f32
  %c2 = arith.constant -128 : i8
  %e3 = tensor.empty() : tensor<12x384x64xi8>
  %q4 = iree_linalg_ext.quantize_affine <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], quant_min = -128, quant_max = 127>
      ins(%a, %c1, %c2 : tensor<12x384x64xf32>, f32, i8) outs(%e3 : tensor<12x384x64xi8>) -> tensor<12x384x64xi8>
  %e5 = tensor.empty() : tensor<12x384x64xf32>
  %dq6 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%q4, %c1, %c2 : tensor<12x384x64xi8>, f32, i8) outs(%e5 : tensor<12x384x64xf32>) -> tensor<12x384x64xf32>
  %c7 = arith.constant 0.04 : f32
  %c8 = arith.constant -128 : i8
  %e9 = tensor.empty() : tensor<12x64x384xi8>
  %q10 = iree_linalg_ext.quantize_affine <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], quant_min = -128, quant_max = 127>
      ins(%b, %c7, %c8 : tensor<12x64x384xf32>, f32, i8) outs(%e9 : tensor<12x64x384xi8>) -> tensor<12x64x384xi8>
  %e11 = tensor.empty() : tensor<12x64x384xf32>
  %dq12 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%q10, %c7, %c8 : tensor<12x64x384xi8>, f32, i8) outs(%e11 : tensor<12x64x384xf32>) -> tensor<12x64x384xf32>
  %e13 = tensor.empty() : tensor<12x384x384xf32>
  %c14 = arith.constant 0.0 : f32
  %f15 = linalg.fill ins(%c14 : f32) outs(%e13 : tensor<12x384x384xf32>) -> tensor<12x384x384xf32>
  %bmm16 = linalg.batch_matmul ins(%dq6, %dq12 : tensor<12x384x64xf32>, tensor<12x64x384xf32>) outs(%f15 : tensor<12x384x384xf32>) -> tensor<12x384x384xf32>
  return %bmm16 : tensor<12x384x384xf32>
}

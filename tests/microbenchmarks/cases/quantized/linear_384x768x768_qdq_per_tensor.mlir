// QDQ linear layer: [384, 768] x [768, 768]. Per-tensor activation quantization
// (zero point -128), i8 weight with per-tensor scale, f32 output.

util.global private @weight = #util.byte_pattern<63> : tensor<768x768xi8>

func.func @linear_384x768x768_qdq_per_tensor(%x: tensor<384x768xf32>) -> tensor<384x768xf32> {
  %c1 = arith.constant 0.05 : f32
  %c2 = arith.constant -128 : i8
  %e3 = tensor.empty() : tensor<384x768xi8>
  %q4 = iree_linalg_ext.quantize_affine <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], quant_min = -128, quant_max = 127>
      ins(%x, %c1, %c2 : tensor<384x768xf32>, f32, i8) outs(%e3 : tensor<384x768xi8>) -> tensor<384x768xi8>
  %e5 = tensor.empty() : tensor<384x768xf32>
  %dq6 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]>
      ins(%q4, %c1, %c2 : tensor<384x768xi8>, f32, i8) outs(%e5 : tensor<384x768xf32>) -> tensor<384x768xf32>
  %weight = util.global.load @weight : tensor<768x768xi8>
  %c7 = arith.constant 0.02 : f32
  %c8 = arith.constant 0 : i8
  %e9 = tensor.empty() : tensor<768x768xf32>
  %wdq10 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]>
      ins(%weight, %c7, %c8 : tensor<768x768xi8>, f32, i8) outs(%e9 : tensor<768x768xf32>) -> tensor<768x768xf32>
  %e11 = tensor.empty() : tensor<384x768xf32>
  %c12 = arith.constant 0.0 : f32
  %f13 = linalg.fill ins(%c12 : f32) outs(%e11 : tensor<384x768xf32>) -> tensor<384x768xf32>
  %mm14 = linalg.matmul ins(%dq6, %wdq10 : tensor<384x768xf32>, tensor<768x768xf32>) outs(%f13 : tensor<384x768xf32>) -> tensor<384x768xf32>
  return %mm14 : tensor<384x768xf32>
}

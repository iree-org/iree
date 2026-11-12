// QDQ linear layer: [384, 768] x [768, 3072] + bias. Per-tensor activation
// quantization (zero point -128), i8 weight with per-channel scale, f32 output.

util.global private @weight = #util.byte_pattern<63> : tensor<768x3072xi8>
util.global private @weight_scale = #util.byte_pattern<63> : tensor<3072xf32>
util.global private @bias = #util.byte_pattern<63> : tensor<3072xf32>

func.func @linear_384x3072x768_bias_qdq_per_channel(%x: tensor<384x768xf32>) -> tensor<384x3072xf32> {
  %c1 = arith.constant 0.05 : f32
  %c2 = arith.constant -128 : i8
  %e3 = tensor.empty() : tensor<384x768xi8>
  %q4 = iree_linalg_ext.quantize_affine <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], quant_min = -128, quant_max = 127>
      ins(%x, %c1, %c2 : tensor<384x768xf32>, f32, i8) outs(%e3 : tensor<384x768xi8>) -> tensor<384x768xi8>
  %e5 = tensor.empty() : tensor<384x768xf32>
  %dq6 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]>
      ins(%q4, %c1, %c2 : tensor<384x768xi8>, f32, i8) outs(%e5 : tensor<384x768xf32>) -> tensor<384x768xf32>
  %weight = util.global.load @weight : tensor<768x3072xi8>
  %weight_scale = util.global.load @weight_scale : tensor<3072xf32>
  %c7 = arith.constant dense<0> : tensor<3072xi8>
  %e8 = tensor.empty() : tensor<768x3072xf32>
  %wdq9 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>]>
      ins(%weight, %weight_scale, %c7 : tensor<768x3072xi8>, tensor<3072xf32>, tensor<3072xi8>) outs(%e8 : tensor<768x3072xf32>) -> tensor<768x3072xf32>
  %bias = util.global.load @bias : tensor<3072xf32>
  %e10 = tensor.empty() : tensor<384x3072xf32>
  %b11 = linalg.broadcast ins(%bias : tensor<3072xf32>) outs(%e10 : tensor<384x3072xf32>) dimensions = [0]
  %mm12 = linalg.matmul ins(%dq6, %wdq9 : tensor<384x768xf32>, tensor<768x3072xf32>) outs(%b11 : tensor<384x3072xf32>) -> tensor<384x3072xf32>
  return %mm12 : tensor<384x3072xf32>
}

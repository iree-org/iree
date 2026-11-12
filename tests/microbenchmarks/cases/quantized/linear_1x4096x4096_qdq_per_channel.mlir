// QDQ linear layer with M = 1: [1, 4096] x [4096, 4096]. Per-tensor activation
// quantization (zero point -128), i8 weight with per-channel scale.

util.global private @weight = #util.byte_pattern<63> : tensor<4096x4096xi8>
util.global private @weight_scale = #util.byte_pattern<63> : tensor<4096xf32>

func.func @linear_1x4096x4096_qdq_per_channel(%x: tensor<1x4096xf32>) -> tensor<1x4096xf32> {
  %c1 = arith.constant 0.05 : f32
  %c2 = arith.constant -128 : i8
  %e3 = tensor.empty() : tensor<1x4096xi8>
  %q4 = iree_linalg_ext.quantize_affine <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], quant_min = -128, quant_max = 127>
      ins(%x, %c1, %c2 : tensor<1x4096xf32>, f32, i8) outs(%e3 : tensor<1x4096xi8>) -> tensor<1x4096xi8>
  %e5 = tensor.empty() : tensor<1x4096xf32>
  %dq6 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]>
      ins(%q4, %c1, %c2 : tensor<1x4096xi8>, f32, i8) outs(%e5 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
  %weight = util.global.load @weight : tensor<4096x4096xi8>
  %weight_scale = util.global.load @weight_scale : tensor<4096xf32>
  %c7 = arith.constant dense<0> : tensor<4096xi8>
  %e8 = tensor.empty() : tensor<4096x4096xf32>
  %wdq9 = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>]>
      ins(%weight, %weight_scale, %c7 : tensor<4096x4096xi8>, tensor<4096xf32>, tensor<4096xi8>) outs(%e8 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  %e10 = tensor.empty() : tensor<1x4096xf32>
  %c11 = arith.constant 0.0 : f32
  %f12 = linalg.fill ins(%c11 : f32) outs(%e10 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
  %mm13 = linalg.matmul ins(%dq6, %wdq9 : tensor<1x4096xf32>, tensor<4096x4096xf32>) outs(%f12 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
  return %mm13 : tensor<1x4096xf32>
}

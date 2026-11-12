// QDQ batch matmul with a weight shared across the batch, BxMxKxN = 8x384x768x512.
// Form A: the constant i8 weight [K, N] (per-channel scale) is dequantized at
// its own rank, then broadcast along the batch.

util.global private @weight = #util.byte_pattern<63> : tensor<768x512xi8>
util.global private @weight_scale = #util.byte_pattern<63> : tensor<512xf32>

func.func @bmm_shared_weight_8x384x768x512_qdq_form_a(%x: tensor<8x384x768xf32>) -> tensor<8x384x512xf32> {
  %sx = arith.constant 0.05 : f32
  %zx = arith.constant -128 : i8
  %xq_init = tensor.empty() : tensor<8x384x768xi8>
  %xq = iree_linalg_ext.quantize_affine <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], quant_min = -128, quant_max = 127>
      ins(%x, %sx, %zx : tensor<8x384x768xf32>, f32, i8) outs(%xq_init : tensor<8x384x768xi8>) -> tensor<8x384x768xi8>
  %xdq_init = tensor.empty() : tensor<8x384x768xf32>
  %xdq = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]>
      ins(%xq, %sx, %zx : tensor<8x384x768xi8>, f32, i8) outs(%xdq_init : tensor<8x384x768xf32>) -> tensor<8x384x768xf32>
  %weight = util.global.load @weight : tensor<768x512xi8>
  %weight_scale = util.global.load @weight_scale : tensor<512xf32>
  %wdq_init = tensor.empty() : tensor<768x512xf32>
  %wdq = iree_linalg_ext.dequantize_affine <indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>]>
      ins(%weight, %weight_scale : tensor<768x512xi8>, tensor<512xf32>) outs(%wdq_init : tensor<768x512xf32>) -> tensor<768x512xf32>
  %wb_init = tensor.empty() : tensor<8x768x512xf32>
  %wb = linalg.broadcast ins(%wdq : tensor<768x512xf32>) outs(%wb_init : tensor<8x768x512xf32>) dimensions = [0]
  %zero = arith.constant 0.0 : f32
  %out_init = tensor.empty() : tensor<8x384x512xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%out_init : tensor<8x384x512xf32>) -> tensor<8x384x512xf32>
  %result = linalg.batch_matmul ins(%xdq, %wb : tensor<8x384x768xf32>, tensor<8x768x512xf32>) outs(%fill : tensor<8x384x512xf32>) -> tensor<8x384x512xf32>
  return %result : tensor<8x384x512xf32>
}

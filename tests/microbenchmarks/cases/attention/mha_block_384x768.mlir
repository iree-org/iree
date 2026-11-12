// Small multi-head attention block: Q/K/V projections with constant weights,
// head split, iree_linalg_ext.attention, head merge, output projection.
// Sequence 384, model dim 768, 12 heads of 64, f32.

util.global private @wq = #util.byte_pattern<63> : tensor<768x768xf32>
util.global private @wk = #util.byte_pattern<63> : tensor<768x768xf32>
util.global private @wv = #util.byte_pattern<63> : tensor<768x768xf32>
util.global private @wo = #util.byte_pattern<63> : tensor<768x768xf32>

func.func @mha_block_384x768(%x: tensor<384x768xf32>) -> tensor<384x768xf32> {
  %wq = util.global.load @wq : tensor<768x768xf32>
  %wk = util.global.load @wk : tensor<768x768xf32>
  %wv = util.global.load @wv : tensor<768x768xf32>
  %wo = util.global.load @wo : tensor<768x768xf32>
  %zero = arith.constant 0.0 : f32
  %scale = arith.constant 0.125 : f32
  %e = tensor.empty() : tensor<384x768xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e : tensor<384x768xf32>) -> tensor<384x768xf32>
  %q = linalg.matmul ins(%x, %wq : tensor<384x768xf32>, tensor<768x768xf32>) outs(%f : tensor<384x768xf32>) -> tensor<384x768xf32>
  %k = linalg.matmul ins(%x, %wk : tensor<384x768xf32>, tensor<768x768xf32>) outs(%f : tensor<384x768xf32>) -> tensor<384x768xf32>
  %v = linalg.matmul ins(%x, %wv : tensor<384x768xf32>, tensor<768x768xf32>) outs(%f : tensor<384x768xf32>) -> tensor<384x768xf32>
  %q3 = tensor.expand_shape %q [[0], [1, 2]] output_shape [384, 12, 64] : tensor<384x768xf32> into tensor<384x12x64xf32>
  %k3 = tensor.expand_shape %k [[0], [1, 2]] output_shape [384, 12, 64] : tensor<384x768xf32> into tensor<384x12x64xf32>
  %v3 = tensor.expand_shape %v [[0], [1, 2]] output_shape [384, 12, 64] : tensor<384x768xf32> into tensor<384x12x64xf32>
  %eh = tensor.empty() : tensor<12x384x64xf32>
  %qh = linalg.transpose ins(%q3 : tensor<384x12x64xf32>) outs(%eh : tensor<12x384x64xf32>) permutation = [1, 0, 2]
  %kh = linalg.transpose ins(%k3 : tensor<384x12x64xf32>) outs(%eh : tensor<12x384x64xf32>) permutation = [1, 0, 2]
  %vh = linalg.transpose ins(%v3 : tensor<384x12x64xf32>) outs(%eh : tensor<12x384x64xf32>) permutation = [1, 0, 2]
  %a = iree_linalg_ext.attention <indexing_maps = [
          affine_map<(b, m, k1, k2, n) -> (b, m, k1)>,
          affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>,
          affine_map<(b, m, k1, k2, n) -> (b, k2, n)>,
          affine_map<(b, m, k1, k2, n) -> ()>,
          affine_map<(b, m, k1, k2, n) -> (b, m, n)>]>
      ins(%qh, %kh, %vh, %scale : tensor<12x384x64xf32>, tensor<12x384x64xf32>, tensor<12x384x64xf32>, f32)
      outs(%eh : tensor<12x384x64xf32>) {
  ^bb0(%s: f32):
    iree_linalg_ext.yield %s : f32
  } -> tensor<12x384x64xf32>
  %et = tensor.empty() : tensor<384x12x64xf32>
  %at = linalg.transpose ins(%a : tensor<12x384x64xf32>) outs(%et : tensor<384x12x64xf32>) permutation = [1, 0, 2]
  %a2 = tensor.collapse_shape %at [[0], [1, 2]] : tensor<384x12x64xf32> into tensor<384x768xf32>
  %o = linalg.matmul ins(%a2, %wo : tensor<384x768xf32>, tensor<768x768xf32>) outs(%f : tensor<384x768xf32>) -> tensor<384x768xf32>
  return %o : tensor<384x768xf32>
}

// iree_linalg_ext.attention, 12 heads, sequence 384, head dim 64, f32.

func.func @attention_op_12x384x64(%q: tensor<12x384x64xf32>, %k: tensor<12x384x64xf32>, %v: tensor<12x384x64xf32>) -> tensor<12x384x64xf32> {
  %scale = arith.constant 0.125 : f32
  %e = tensor.empty() : tensor<12x384x64xf32>
  %r = iree_linalg_ext.attention <indexing_maps = [
          affine_map<(b, m, k1, k2, n) -> (b, m, k1)>,
          affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>,
          affine_map<(b, m, k1, k2, n) -> (b, k2, n)>,
          affine_map<(b, m, k1, k2, n) -> ()>,
          affine_map<(b, m, k1, k2, n) -> (b, m, n)>]>
      ins(%q, %k, %v, %scale : tensor<12x384x64xf32>, tensor<12x384x64xf32>, tensor<12x384x64xf32>, f32)
      outs(%e : tensor<12x384x64xf32>) {
  ^bb0(%s: f32):
    iree_linalg_ext.yield %s : f32
  } -> tensor<12x384x64xf32>
  return %r : tensor<12x384x64xf32>
}

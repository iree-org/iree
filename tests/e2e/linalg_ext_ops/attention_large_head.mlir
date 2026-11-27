// A head of 144 is above the typical 128: its scores still reduce over the
// whole head before the softmax. Every lane of key j and of value j holds j,
// the query is all ones and the scale is 1/144, so key j scores exactly j and
// each output element is sum(j * e^j) / sum(e^j) over j = 0..3 = 2.492653.
// Reducing only the first 128 lanes would score key j as (128/144) * j and
// give 2.419505.
func.func @attention_head_144() {
  %query = util.unfoldable_constant dense<1.0> : tensor<1x1x144xf32>
  %per_key = util.unfoldable_constant dense<[0.0, 1.0, 2.0, 3.0]> : tensor<4xf32>
  %key_init = tensor.empty() : tensor<1x4x144xf32>
  %key = linalg.broadcast ins(%per_key : tensor<4xf32>) outs(%key_init : tensor<1x4x144xf32>) dimensions = [0, 2]
  %value_init = tensor.empty() : tensor<1x4x6xf32>
  %value = linalg.broadcast ins(%per_key : tensor<4xf32>) outs(%value_init : tensor<1x4x6xf32>) dimensions = [0, 2]
  %scale = arith.constant 0.00694444444 : f32
  %init = tensor.empty() : tensor<1x1x6xf32>
  %result = iree_linalg_ext.attention
      <indexing_maps = [affine_map<(b, m, k1, k2, n) -> (b, m, k1)>,
                        affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>,
                        affine_map<(b, m, k1, k2, n) -> (b, k2, n)>,
                        affine_map<(b, m, k1, k2, n) -> ()>,
                        affine_map<(b, m, k1, k2, n) -> (b, m, n)>]>
      ins(%query, %key, %value, %scale : tensor<1x1x144xf32>, tensor<1x4x144xf32>, tensor<1x4x6xf32>, f32)
      outs(%init : tensor<1x1x6xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<1x1x6xf32>
  check.expect_almost_eq_const(%result, dense<2.492653> : tensor<1x1x6xf32>) : tensor<1x1x6xf32>
  return
}

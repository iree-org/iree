// Keys [0, 5) and [16, 19) are visible in every row; the excluded keys 5, 6, 7
// and 20 share K/V tiles with visible keys and hold non-finite V.
//
// Every query is 1.0 and the keys are 0.0 unless a case says otherwise, so all
// visible scores are equal and each output element is the mean of the visible
// values. The visible keys hold values 0..7 (keys 0..4 hold 0..4 and keys
// 16..18 hold 5..7), whose mean is 3.5.

// Excluded NaN and Inf in V do not reach the output.
func.func @excluded_non_finite_values() {
  %key = util.unfoldable_constant dense<0.0> : tensor<1x24x4xf32>
  %values = util.unfoldable_constant dense<[
      0.0, 1.0, 2.0, 3.0, 4.0, 0x7FC00000, 0x7F800000, 0x7FC00000,
      0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
      5.0, 6.0, 7.0, 100.0, 0x7F800000, 100.0, 100.0, 100.0]> : tensor<24xf32>
  %result = func.call @attention_shared_mask(%key, %values)
      : (tensor<1x24x4xf32>, tensor<24xf32>) -> tensor<1x2x5xf32>
  check.expect_almost_eq_const(%result, dense<3.5> : tensor<1x2x5xf32>) : tensor<1x2x5xf32>
  return
}

// A visible NaN in V reaches every row that sees it. Key 2 holds NaN.
func.func @visible_nan_value() {
  %key = util.unfoldable_constant dense<0.0> : tensor<1x24x4xf32>
  %values = util.unfoldable_constant dense<[
      0.0, 1.0, 0x7FC00000, 3.0, 4.0, 0x7FC00000, 0x7F800000, 0x7FC00000,
      0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
      5.0, 6.0, 7.0, 100.0, 0x7F800000, 100.0, 100.0, 100.0]> : tensor<24xf32>
  %result = func.call @attention_shared_mask(%key, %values)
      : (tensor<1x24x4xf32>, tensor<24xf32>) -> tensor<1x2x5xf32>
  check.expect_almost_eq_const(%result, dense<0x7FC00000> : tensor<1x2x5xf32>) : tensor<1x2x5xf32>
  return
}

// A visible key whose probability underflows to 0 still contributes 0 * Inf.
// Key 3 holds -10000.0 in every lane, so its score is 0.5 * 4 * -10000.0 and
// its probability is 0; its value is Inf.
func.func @visible_zero_times_inf() {
  %zero_keys = util.unfoldable_constant dense<0.0> : tensor<1x3x4xf32>
  %low_key = util.unfoldable_constant dense<-10000.0> : tensor<1x1x4xf32>
  %more_zero_keys = util.unfoldable_constant dense<0.0> : tensor<1x20x4xf32>
  %key = tensor.concat dim(1) %zero_keys, %low_key, %more_zero_keys
      : (tensor<1x3x4xf32>, tensor<1x1x4xf32>, tensor<1x20x4xf32>) -> tensor<1x24x4xf32>
  %values = util.unfoldable_constant dense<[
      0.0, 1.0, 2.0, 0x7F800000, 4.0, 0x7FC00000, 0x7F800000, 0x7FC00000,
      0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000,
      5.0, 6.0, 7.0, 100.0, 0x7F800000, 100.0, 100.0, 100.0]> : tensor<24xf32>
  %result = func.call @attention_shared_mask(%key, %values)
      : (tensor<1x24x4xf32>, tensor<24xf32>) -> tensor<1x2x5xf32>
  check.expect_almost_eq_const(%result, dense<0x7FC00000> : tensor<1x2x5xf32>) : tensor<1x2x5xf32>
  return
}

// Attention of two all-ones query rows over 24 keys, where key k holds
// `values[k]` in every lane and the mask shared by both rows shows keys
// [0, 5) and [16, 19).
func.func private @attention_shared_mask(%key: tensor<1x24x4xf32>, %values: tensor<24xf32>) -> tensor<1x2x5xf32> {
  %query = util.unfoldable_constant dense<1.0> : tensor<1x2x4xf32>
  %value_init = tensor.empty() : tensor<1x24x5xf32>
  %value = linalg.broadcast ins(%values : tensor<24xf32>) outs(%value_init : tensor<1x24x5xf32>) dimensions = [0, 2]
  %mask = util.unfoldable_constant dense<[[
      true, true, true, true, true, false, false, false,
      false, false, false, false, false, false, false, false,
      true, true, true, false, false, false, false, false]]> : tensor<1x24xi1>
  %scale = arith.constant 0.5 : f32
  %init = tensor.empty() : tensor<1x2x5xf32>
  %result = iree_linalg_ext.attention
      <indexing_maps = [affine_map<(b, m, k1, k2, n) -> (b, m, k1)>,
                        affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>,
                        affine_map<(b, m, k1, k2, n) -> (b, k2, n)>,
                        affine_map<(b, m, k1, k2, n) -> ()>,
                        affine_map<(b, m, k1, k2, n) -> (b, k2)>,
                        affine_map<(b, m, k1, k2, n) -> (b, m, n)>]>
      ins(%query, %key, %value, %scale, %mask : tensor<1x2x4xf32>, tensor<1x24x4xf32>, tensor<1x24x5xf32>, f32, tensor<1x24xi1>)
      outs(%init : tensor<1x2x5xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<1x2x5xf32>
  return %result : tensor<1x2x5xf32>
}

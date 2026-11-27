// Keys [0, 16) and [32, 48) are hidden in every row, so they fill whole K/V
// tiles of up to 16 keys; keys [16, 32) share tiles with visible keys.
//
// Every query is 1.0 and every visible key is 0.0, so all visible scores are
// equal and a row's output is the mean of the values it sees. Value row k of
// the middle block holds k - 16 in every lane (see @values_around_middle).
// A value from a hidden key that leaks into a row changes that mean or makes
// it NaN.

// Non-finite K and V at hidden keys do not reach visible rows. Row 0 sees the
// even keys in [16, 32), whose values average to 7.0; row 1 sees keys
// [28, 32), whose values average to 13.5. Key 23 shares a tile with visible
// keys, is hidden from both rows and holds NaN.
func.func @bool_mask_prefix_suffix() {
  %nan_keys = util.unfoldable_constant dense<0x7FC00000> : tensor<1x16x4xf32>
  %zero_keys = util.unfoldable_constant dense<0.0> : tensor<1x7x4xf32>
  %nan_key = util.unfoldable_constant dense<0x7FC00000> : tensor<1x1x4xf32>
  %more_zero_keys = util.unfoldable_constant dense<0.0> : tensor<1x8x4xf32>
  %inf_keys = util.unfoldable_constant dense<0x7F800000> : tensor<1x16x4xf32>
  %key = tensor.concat dim(1) %nan_keys, %zero_keys, %nan_key, %more_zero_keys, %inf_keys
      : (tensor<1x16x4xf32>, tensor<1x7x4xf32>, tensor<1x1x4xf32>, tensor<1x8x4xf32>, tensor<1x16x4xf32>) -> tensor<1x48x4xf32>
  %nan_values = util.unfoldable_constant dense<0x7FC00000> : tensor<1x16x5xf32>
  %inf_values = util.unfoldable_constant dense<0x7F800000> : tensor<1x16x5xf32>
  %value = func.call @values_around_middle(%nan_values, %inf_values)
      : (tensor<1x16x5xf32>, tensor<1x16x5xf32>) -> tensor<1x48x5xf32>
  %hidden = util.unfoldable_constant dense<false> : tensor<1x2x16xi1>
  %middle = util.unfoldable_constant dense<[[
      [true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false],
      [false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true]]]> : tensor<1x2x16xi1>
  %mask = tensor.concat dim(2) %hidden, %middle, %hidden
      : (tensor<1x2x16xi1>, tensor<1x2x16xi1>, tensor<1x2x16xi1>) -> tensor<1x2x48xi1>
  %result = func.call @attention_row_mask_i1(%key, %value, %mask)
      : (tensor<1x48x4xf32>, tensor<1x48x5xf32>, tensor<1x2x48xi1>) -> tensor<1x2x5xf32>
  check.expect_almost_eq_const(%result, dense<[[
      [7.0, 7.0, 7.0, 7.0, 7.0],
      [13.5, 13.5, 13.5, 13.5, 13.5]]]> : tensor<1x2x5xf32>) : tensor<1x2x5xf32>
  return
}

// A -inf entry excludes its key, even one whose score is NaN; a finite
// negative entry is a bias. Row 0 has bias 0 on every middle key, so it
// averages values 0..15 to 7.5. Row 1 has bias ln(1/2) on keys [16, 24), which
// halves their weight: (0.5 * (0 + ... + 7) + (8 + ... + 15)) / 12 = 8.833333.
func.func @additive_mask() {
  %nan_keys = util.unfoldable_constant dense<0x7FC00000> : tensor<1x16x4xf32>
  %zero_keys = util.unfoldable_constant dense<0.0> : tensor<1x32x4xf32>
  %key = tensor.concat dim(1) %nan_keys, %zero_keys
      : (tensor<1x16x4xf32>, tensor<1x32x4xf32>) -> tensor<1x48x4xf32>
  %hidden_values = util.unfoldable_constant dense<100.0> : tensor<1x16x5xf32>
  %value = func.call @values_around_middle(%hidden_values, %hidden_values)
      : (tensor<1x16x5xf32>, tensor<1x16x5xf32>) -> tensor<1x48x5xf32>
  %hidden = util.unfoldable_constant dense<0xFF800000> : tensor<1x2x16xf32>
  %middle = util.unfoldable_constant dense<[[
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [-0.6931472, -0.6931472, -0.6931472, -0.6931472, -0.6931472, -0.6931472, -0.6931472, -0.6931472,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]> : tensor<1x2x16xf32>
  %mask = tensor.concat dim(2) %hidden, %middle, %hidden
      : (tensor<1x2x16xf32>, tensor<1x2x16xf32>, tensor<1x2x16xf32>) -> tensor<1x2x48xf32>
  %result = func.call @attention_row_mask_f32(%key, %value, %mask)
      : (tensor<1x48x4xf32>, tensor<1x48x5xf32>, tensor<1x2x48xf32>) -> tensor<1x2x5xf32>
  check.expect_almost_eq_const(%result, dense<[[
      [7.5, 7.5, 7.5, 7.5, 7.5],
      [8.833333, 8.833333, 8.833333, 8.833333, 8.833333]]]> : tensor<1x2x5xf32>) : tensor<1x2x5xf32>
  return
}

// A mask shared by every query skips the same tiles. Both rows see the even
// keys in [16, 32), whose values average to 7.0.
func.func @broadcast_mask() {
  %zero_keys = util.unfoldable_constant dense<0.0> : tensor<1x48x4xf32>
  %nan_values = util.unfoldable_constant dense<0x7FC00000> : tensor<1x16x5xf32>
  %value = func.call @values_around_middle(%nan_values, %nan_values)
      : (tensor<1x16x5xf32>, tensor<1x16x5xf32>) -> tensor<1x48x5xf32>
  %hidden = util.unfoldable_constant dense<false> : tensor<1x16xi1>
  %middle = util.unfoldable_constant dense<[
      [true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false]]> : tensor<1x16xi1>
  %mask = tensor.concat dim(1) %hidden, %middle, %hidden
      : (tensor<1x16xi1>, tensor<1x16xi1>, tensor<1x16xi1>) -> tensor<1x48xi1>
  %query = util.unfoldable_constant dense<1.0> : tensor<1x2x4xf32>
  %scale = arith.constant 0.5 : f32
  %init = tensor.empty() : tensor<1x2x5xf32>
  %result = iree_linalg_ext.attention
      <indexing_maps = [affine_map<(b, m, k1, k2, n) -> (b, m, k1)>,
                        affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>,
                        affine_map<(b, m, k1, k2, n) -> (b, k2, n)>,
                        affine_map<(b, m, k1, k2, n) -> ()>,
                        affine_map<(b, m, k1, k2, n) -> (b, k2)>,
                        affine_map<(b, m, k1, k2, n) -> (b, m, n)>]>
      ins(%query, %zero_keys, %value, %scale, %mask : tensor<1x2x4xf32>, tensor<1x48x4xf32>, tensor<1x48x5xf32>, f32, tensor<1x48xi1>)
      outs(%init : tensor<1x2x5xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<1x2x5xf32>
  check.expect_almost_eq_const(%result, dense<7.0> : tensor<1x2x5xf32>) : tensor<1x2x5xf32>
  return
}

// A NaN mask entry is added to its score, so its row is NaN. Only row 1 has
// one, at key 26.
func.func @nan_mask_entry_propagates() {
  %zero_keys = util.unfoldable_constant dense<0.0> : tensor<1x48x4xf32>
  %hidden_values = util.unfoldable_constant dense<100.0> : tensor<1x16x5xf32>
  %value = func.call @values_around_middle(%hidden_values, %hidden_values)
      : (tensor<1x16x5xf32>, tensor<1x16x5xf32>) -> tensor<1x48x5xf32>
  %hidden = util.unfoldable_constant dense<0xFF800000> : tensor<1x2x16xf32>
  %middle = util.unfoldable_constant dense<[[
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0x7FC00000, 0.0, 0.0, 0.0, 0.0, 0.0]]]> : tensor<1x2x16xf32>
  %mask = tensor.concat dim(2) %hidden, %middle, %hidden
      : (tensor<1x2x16xf32>, tensor<1x2x16xf32>, tensor<1x2x16xf32>) -> tensor<1x2x48xf32>
  %result = func.call @attention_row_mask_f32(%zero_keys, %value, %mask)
      : (tensor<1x48x4xf32>, tensor<1x48x5xf32>, tensor<1x2x48xf32>) -> tensor<1x2x5xf32>
  %nan_rows = func.call @nan_rows(%result) : (tensor<1x2x5xf32>) -> tensor<1x2xi32>
  check.expect_eq_const(%nan_rows, dense<[[0, 1]]> : tensor<1x2xi32>) : tensor<1x2xi32>
  return
}

// A NaN key that some row sees makes that row NaN. Key 31 holds NaN; row 1
// sees it, row 0 sees only even keys.
func.func @visible_nan_propagates() {
  %zero_keys = util.unfoldable_constant dense<0.0> : tensor<1x31x4xf32>
  %nan_key = util.unfoldable_constant dense<0x7FC00000> : tensor<1x1x4xf32>
  %more_zero_keys = util.unfoldable_constant dense<0.0> : tensor<1x16x4xf32>
  %key = tensor.concat dim(1) %zero_keys, %nan_key, %more_zero_keys
      : (tensor<1x31x4xf32>, tensor<1x1x4xf32>, tensor<1x16x4xf32>) -> tensor<1x48x4xf32>
  %hidden_values = util.unfoldable_constant dense<100.0> : tensor<1x16x5xf32>
  %value = func.call @values_around_middle(%hidden_values, %hidden_values)
      : (tensor<1x16x5xf32>, tensor<1x16x5xf32>) -> tensor<1x48x5xf32>
  %hidden = util.unfoldable_constant dense<false> : tensor<1x2x16xi1>
  %middle = util.unfoldable_constant dense<[[
      [true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false],
      [false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true]]]> : tensor<1x2x16xi1>
  %mask = tensor.concat dim(2) %hidden, %middle, %hidden
      : (tensor<1x2x16xi1>, tensor<1x2x16xi1>, tensor<1x2x16xi1>) -> tensor<1x2x48xi1>
  %result = func.call @attention_row_mask_i1(%key, %value, %mask)
      : (tensor<1x48x4xf32>, tensor<1x48x5xf32>, tensor<1x2x48xi1>) -> tensor<1x2x5xf32>
  %nan_rows = func.call @nan_rows(%result) : (tensor<1x2x5xf32>) -> tensor<1x2xi32>
  check.expect_eq_const(%nan_rows, dense<[[0, 1]]> : tensor<1x2xi32>) : tensor<1x2xi32>
  return
}

// Values for 48 keys: `first` for keys [0, 16), k - 16 in every lane of key k
// in [16, 32), and `last` for keys [32, 48).
func.func private @values_around_middle(%first: tensor<1x16x5xf32>, %last: tensor<1x16x5xf32>) -> tensor<1x48x5xf32> {
  %per_key = util.unfoldable_constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]> : tensor<16xf32>
  %init = tensor.empty() : tensor<1x16x5xf32>
  %middle = linalg.broadcast ins(%per_key : tensor<16xf32>) outs(%init : tensor<1x16x5xf32>) dimensions = [0, 2]
  %value = tensor.concat dim(1) %first, %middle, %last
      : (tensor<1x16x5xf32>, tensor<1x16x5xf32>, tensor<1x16x5xf32>) -> tensor<1x48x5xf32>
  return %value : tensor<1x48x5xf32>
}

// Attention of two all-ones query rows with a per-row boolean mask.
func.func private @attention_row_mask_i1(%key: tensor<1x48x4xf32>, %value: tensor<1x48x5xf32>, %mask: tensor<1x2x48xi1>) -> tensor<1x2x5xf32> {
  %query = util.unfoldable_constant dense<1.0> : tensor<1x2x4xf32>
  %scale = arith.constant 0.5 : f32
  %init = tensor.empty() : tensor<1x2x5xf32>
  %result = iree_linalg_ext.attention
      <indexing_maps = [affine_map<(b, m, k1, k2, n) -> (b, m, k1)>,
                        affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>,
                        affine_map<(b, m, k1, k2, n) -> (b, k2, n)>,
                        affine_map<(b, m, k1, k2, n) -> ()>,
                        affine_map<(b, m, k1, k2, n) -> (b, m, k2)>,
                        affine_map<(b, m, k1, k2, n) -> (b, m, n)>]>
      ins(%query, %key, %value, %scale, %mask : tensor<1x2x4xf32>, tensor<1x48x4xf32>, tensor<1x48x5xf32>, f32, tensor<1x2x48xi1>)
      outs(%init : tensor<1x2x5xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<1x2x5xf32>
  return %result : tensor<1x2x5xf32>
}

// Attention of two all-ones query rows with a per-row additive mask.
func.func private @attention_row_mask_f32(%key: tensor<1x48x4xf32>, %value: tensor<1x48x5xf32>, %mask: tensor<1x2x48xf32>) -> tensor<1x2x5xf32> {
  %query = util.unfoldable_constant dense<1.0> : tensor<1x2x4xf32>
  %scale = arith.constant 0.5 : f32
  %init = tensor.empty() : tensor<1x2x5xf32>
  %result = iree_linalg_ext.attention
      <indexing_maps = [affine_map<(b, m, k1, k2, n) -> (b, m, k1)>,
                        affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>,
                        affine_map<(b, m, k1, k2, n) -> (b, k2, n)>,
                        affine_map<(b, m, k1, k2, n) -> ()>,
                        affine_map<(b, m, k1, k2, n) -> (b, m, k2)>,
                        affine_map<(b, m, k1, k2, n) -> (b, m, n)>]>
      ins(%query, %key, %value, %scale, %mask : tensor<1x2x4xf32>, tensor<1x48x4xf32>, tensor<1x48x5xf32>, f32, tensor<1x2x48xf32>)
      outs(%init : tensor<1x2x5xf32>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
  } -> tensor<1x2x5xf32>
  return %result : tensor<1x2x5xf32>
}

// Returns 1 for each output row that holds a NaN, else 0.
func.func private @nan_rows(%result: tensor<1x2x5xf32>) -> tensor<1x2xi32> {
  %zero = arith.constant 0 : i32
  %init = tensor.empty() : tensor<1x2xi32>
  %zeros = linalg.fill ins(%zero : i32) outs(%init : tensor<1x2xi32>) -> tensor<1x2xi32>
  %rows = linalg.generic {
      indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>, affine_map<(b, m, n) -> (b, m)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%result : tensor<1x2x5xf32>) outs(%zeros : tensor<1x2xi32>) {
    ^bb0(%x: f32, %seen: i32):
      %nan = arith.cmpf uno, %x, %x : f32
      %one = arith.extui %nan : i1 to i32
      %any = arith.maxsi %one, %seen : i32
      linalg.yield %any : i32
  } -> tensor<1x2xi32>
  return %rows : tensor<1x2xi32>
}

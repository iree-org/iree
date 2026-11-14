// Operations on tensors with a zero-size dimension. A contraction over an empty
// dimension gives its initial value; the other results are empty or unchanged.

func.func @dot_zero_contraction() {
  %a = util.unfoldable_constant dense<> : tensor<3x0xf32>
  %b = util.unfoldable_constant dense<> : tensor<0x4xf32>
  %r = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<3x0xf32>, tensor<0x4xf32>) -> tensor<3x4xf32>
  check.expect_eq_const(%r, dense<0.0> : tensor<3x4xf32>) : tensor<3x4xf32>
  return
}
func.func @reduce_zero_extent() {
  %a = util.unfoldable_constant dense<> : tensor<3x0xf32>
  %init = stablehlo.constant dense<5.0> : tensor<f32>
  %r = stablehlo.reduce(%a init: %init) applies stablehlo.add across dimensions = [1] : (tensor<3x0xf32>, tensor<f32>) -> tensor<3xf32>
  check.expect_eq_const(%r, dense<5.0> : tensor<3xf32>) : tensor<3xf32>
  return
}
func.func @pad_zero_operand() {
  %a = util.unfoldable_constant dense<> : tensor<0x3xf32>
  %v = stablehlo.constant dense<7.0> : tensor<f32>
  %r = stablehlo.pad %a, %v, low = [1, 0], high = [1, 1], interior = [0, 0] : (tensor<0x3xf32>, tensor<f32>) -> tensor<2x4xf32>
  check.expect_eq_const(%r, dense<7.0> : tensor<2x4xf32>) : tensor<2x4xf32>
  return
}
func.func @concat_zero_operand() {
  %a = util.unfoldable_constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %z = util.unfoldable_constant dense<> : tensor<2x0xf32>
  %b = util.unfoldable_constant dense<[[5.0], [6.0]]> : tensor<2x1xf32>
  %r = stablehlo.concatenate %a, %z, %b, dim = 1 : (tensor<2x2xf32>, tensor<2x0xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  check.expect_eq_const(%r, dense<[[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]> : tensor<2x3xf32>) : tensor<2x3xf32>
  return
}
func.func @conv_zero_input_feature() {
  %x = util.unfoldable_constant dense<> : tensor<1x4x3x0xf32>
  %w = util.unfoldable_constant dense<> : tensor<2x1x0x5xf32>
  %r = stablehlo.convolution(%x, %w) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x4x3x0xf32>, tensor<2x1x0x5xf32>) -> tensor<1x3x3x5xf32>
  check.expect_eq_const(%r, dense<0.0> : tensor<1x3x3x5xf32>) : tensor<1x3x3x5xf32>
  return
}
func.func @reduce_window_zero_window_count() {
  %a = util.unfoldable_constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %init = stablehlo.constant dense<0.0> : tensor<f32>
  %r = "stablehlo.reduce_window"(%a, %init) ({
  ^bb0(%x: tensor<f32>, %y: tensor<f32>):
    %s = stablehlo.add %x, %y : tensor<f32>
    stablehlo.return %s : tensor<f32>
  }) {window_dimensions = array<i64: 3, 1>} : (tensor<2x2xf32>, tensor<f32>) -> tensor<0x2xf32>
  check.expect_eq_const(%r, dense<> : tensor<0x2xf32>) : tensor<0x2xf32>
  return
}
func.func @gather_zero_batch() {
  %a = util.unfoldable_constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>
  %i = util.unfoldable_constant dense<> : tensor<0x1xi32>
  %r = "stablehlo.gather"(%a, %i) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 2>} : (tensor<3x2xf32>, tensor<0x1xi32>) -> tensor<0x2xf32>
  check.expect_eq_const(%r, dense<> : tensor<0x2xf32>) : tensor<0x2xf32>
  return
}
func.func @sort_zero() {
  %a = util.unfoldable_constant dense<> : tensor<2x0xf32>
  %r = "stablehlo.sort"(%a) ({
  ^bb0(%x: tensor<f32>, %y: tensor<f32>):
    %c = stablehlo.compare LT, %x, %y : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  }) {dimension = 1 : i64} : (tensor<2x0xf32>) -> tensor<2x0xf32>
  check.expect_eq_const(%r, dense<> : tensor<2x0xf32>) : tensor<2x0xf32>
  return
}
func.func @transpose_reverse_broadcast_zero() {
  %a = util.unfoldable_constant dense<> : tensor<0x3xf32>
  %t = stablehlo.transpose %a, dims = [1, 0] : (tensor<0x3xf32>) -> tensor<3x0xf32>
  %v = stablehlo.reverse %t, dims = [0] : tensor<3x0xf32>
  %b = stablehlo.broadcast_in_dim %v, dims = [1, 2] : (tensor<3x0xf32>) -> tensor<2x3x0xf32>
  check.expect_eq_const(%b, dense<> : tensor<2x3x0xf32>) : tensor<2x3x0xf32>
  %io = stablehlo.iota dim = 0 : tensor<0x4xi32>
  check.expect_eq_const(%io, dense<> : tensor<0x4xi32>) : tensor<0x4xi32>
  return
}
func.func @dynamic_slice_zero_size() {
  %a = util.unfoldable_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %i = util.unfoldable_constant dense<1> : tensor<i32>
  %r = stablehlo.dynamic_slice %a, %i, %i, sizes = [0, 2] : (tensor<2x3xf32>, tensor<i32>, tensor<i32>) -> tensor<0x2xf32>
  check.expect_eq_const(%r, dense<> : tensor<0x2xf32>) : tensor<0x2xf32>
  %u = util.unfoldable_constant dense<> : tensor<0x2xf32>
  %w = stablehlo.dynamic_update_slice %a, %u, %i, %i : (tensor<2x3xf32>, tensor<0x2xf32>, tensor<i32>, tensor<i32>) -> tensor<2x3xf32>
  check.expect_eq_const(%w, dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>) : tensor<2x3xf32>
  return
}

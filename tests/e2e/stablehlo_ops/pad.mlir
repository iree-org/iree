func.func @pad_test() {
  %input = util.unfoldable_constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %c0 = arith.constant dense<0> : tensor<i32>
  %res = "stablehlo.pad"(%input, %c0) {
    edge_padding_low = array<i64: 0, 1>,
    edge_padding_high = array<i64: 1, 5>,
    interior_padding = array<i64: 0, 0>
  } : (tensor<2x3xi32>, tensor<i32>) -> tensor<3x9xi32>
  check.expect_eq_const(%res, dense<[
      [0, 1, 2, 3, 0, 0, 0, 0, 0],
      [0, 4, 5, 6, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0]]> : tensor<3x9xi32>) : tensor<3x9xi32>
  return
}

func.func @pad_no_op() {
  %input = util.unfoldable_constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %c0 = arith.constant dense<0> : tensor<i32>
  %res = "stablehlo.pad"(%input, %c0) {edge_padding_high = array<i64: 0, 0>, edge_padding_low = array<i64: 0, 0>, interior_padding = array<i64: 0, 0>} : (tensor<2x3xi32>, tensor<i32>) -> tensor<2x3xi32>
  check.expect_eq(%res, %input) : tensor<2x3xi32>
  return
}

func.func @pad_static_extreme_crop() {
  %input = util.unfoldable_constant dense<[1, 2, 3]> : tensor<3xi32>
  %pad = stablehlo.constant dense<7> : tensor<i32>
  %r = stablehlo.pad %input, %pad, low = [-9223372036854775804], high = [-9223372036854775806], interior = [9223372036854775805] : (tensor<3xi32>, tensor<i32>) -> tensor<3xi32>
  check.expect_eq_const(%r, dense<[7, 7, 2]> : tensor<3xi32>) : tensor<3xi32>
  return
}

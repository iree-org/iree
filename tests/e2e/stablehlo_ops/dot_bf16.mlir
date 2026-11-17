func.func @bf16() {
  %lhs = util.unfoldable_constant dense<[
    [15.0, 14.0, 13.0],
    [12.0, 11.0, 10.0],
    [09.0, 08.0, 07.0],
    [06.0, 05.0, 04.0],
    [03.0, 02.0, 01.0]]> : tensor<5x3xbf16>
  %rhs = util.unfoldable_constant dense<[
    [15.0, 14.0, 13.0, 12.0, 11.0],
    [10.0, 09.0, 08.0, 07.0, 06.0],
    [05.0, 04.0, 03.0, 02.0, 01.0]]> : tensor<3x5xbf16>
  %res = "stablehlo.dot"(%lhs, %rhs) : (tensor<5x3xbf16>, tensor<3x5xbf16>) -> tensor<5x5xf32>
  check.expect_almost_eq_const(%res, dense<[
    [430.0, 388.0, 346.0, 304.0, 262.0],
    [340.0, 307.0, 274.0, 241.0, 208.0],
    [250.0, 226.0, 202.0, 178.0, 154.0],
    [160.0, 145.0, 130.0, 115.0, 100.0],
    [70.0, 64.0, 58.0, 52.0, 46.0]]> : tensor<5x5xf32>) : tensor<5x5xf32>
  return
}

// Partial BF16 sums round away both ones before cancellation.
func.func @bf16_accumulates_f32() {
  %lhs = util.unfoldable_constant dense<[[256.0, 1.0, 1.0, -256.0]]> : tensor<1x4xbf16>
  %rhs = util.unfoldable_constant dense<1.0> : tensor<4x1xbf16>
  %res = stablehlo.dot %lhs, %rhs, precision = [HIGHEST, HIGHEST] : (tensor<1x4xbf16>, tensor<4x1xbf16>) -> tensor<1x1xbf16>
  check.expect_eq_const(%res, dense<2.0> : tensor<1x1xbf16>) : tensor<1x1xbf16>
  return
}

func.func @bf16_dot_general_accumulates_f32() {
  %lhs = util.unfoldable_constant dense<[[[256.0, 1.0, 1.0, -256.0]], [[256.0, 1.0, 1.0, -256.0]]]> : tensor<2x1x4xbf16>
  %rhs = util.unfoldable_constant dense<1.0> : tensor<2x4x1xbf16>
  %res = stablehlo.dot_general %lhs, %rhs, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x1x4xbf16>, tensor<2x4x1xbf16>) -> tensor<2x1x1xbf16>
  check.expect_eq_const(%res, dense<2.0> : tensor<2x1x1xbf16>) : tensor<2x1x1xbf16>
  return
}

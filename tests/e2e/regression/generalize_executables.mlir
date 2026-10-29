// Tests execution after structurally identical executables have been
// generalized and deduplicated: first two that differ only in an f32
// constant, then two that read a global at different static offsets. In each
// pair the two entry points must pass different values to the resulting
// dispatch.

#map = affine_map<(d0) -> (d0)>

func.func @scale_half() {
  %input = util.unfoldable_constant dense<[2.0, 4.0, 6.0, 8.0]>
      : tensor<4xf32>
  %scale = arith.constant 5.000000e-01 : f32
  %init = tensor.empty() : tensor<4xf32>
  %result = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel"]}
      ins(%input : tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %scaled = arith.mulf %in, %scale : f32
      linalg.yield %scaled : f32
  } -> tensor<4xf32>
  check.expect_almost_eq_const(
      %result, dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>)
      : tensor<4xf32>
  return
}

func.func @scale_quarter() {
  %input = util.unfoldable_constant dense<[2.0, 4.0, 6.0, 8.0]>
      : tensor<4xf32>
  %scale = arith.constant 2.500000e-01 : f32
  %init = tensor.empty() : tensor<4xf32>
  %result = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel"]}
      ins(%input : tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %scaled = arith.mulf %in, %scale : f32
      linalg.yield %scaled : f32
  } -> tensor<4xf32>
  check.expect_almost_eq_const(
      %result, dense<[0.5, 1.0, 1.5, 2.0]> : tensor<4xf32>)
      : tensor<4xf32>
  return
}

// Two entry points that read different rows of the same global at static
// offsets: the offset is promoted to an operand of one shared dispatch.
util.global private @cache : tensor<8x4xf32>
util.initializer {
  %value = util.unfoldable_constant dense<[
      [0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0],
      [8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0],
      [16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0],
      [24.0, 25.0, 26.0, 27.0], [28.0, 29.0, 30.0, 31.0]]> : tensor<8x4xf32>
  util.global.store %value, @cache : tensor<8x4xf32>
  util.return
}

func.func @row_2() {
  %cache = util.global.load @cache : tensor<8x4xf32>
  %row = tensor.extract_slice %cache[2, 0] [1, 4] [1, 1]
      : tensor<8x4xf32> to tensor<4xf32>
  %two = util.unfoldable_constant 2.000000e+00 : f32
  %init = tensor.empty() : tensor<4xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%row : tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %doubled = arith.mulf %in, %two : f32
      linalg.yield %doubled : f32
  } -> tensor<4xf32>
  check.expect_almost_eq_const(
      %result, dense<[16.0, 18.0, 20.0, 22.0]> : tensor<4xf32>)
      : tensor<4xf32>
  return
}

func.func @row_6() {
  %cache = util.global.load @cache : tensor<8x4xf32>
  %row = tensor.extract_slice %cache[6, 0] [1, 4] [1, 1]
      : tensor<8x4xf32> to tensor<4xf32>
  %two = util.unfoldable_constant 2.000000e+00 : f32
  %init = tensor.empty() : tensor<4xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%row : tensor<4xf32>) outs(%init : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %doubled = arith.mulf %in, %two : f32
      linalg.yield %doubled : f32
  } -> tensor<4xf32>
  check.expect_almost_eq_const(
      %result, dense<[48.0, 50.0, 52.0, 54.0]> : tensor<4xf32>)
      : tensor<4xf32>
  return
}

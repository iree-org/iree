// Tests execution after structurally identical executables that differ only
// in an f32 constant have been generalized and deduplicated. The two entry
// points must pass different values to the resulting dispatch.

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

#map0 = affine_map<(d1) -> (d1)>
#map1 = affine_map<(d1) -> (0)>

func.func @demote() {
  %input = util.unfoldable_constant dense<3.0> : tensor<8388608xf32>
  %cst_0 = arith.constant 0.000000e+00 : f64
  %init = tensor.empty() : tensor<1xf64>
  %zeros = linalg.fill ins(%cst_0 : f64) outs(%init : tensor<1xf64>) -> tensor<1xf64>
  %accum = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction"]} ins(%input : tensor<8388608xf32>) outs(%init : tensor<1xf64>) {
  ^bb0(%arg1: f32, %arg2: f64):
    %ext = arith.extf %arg1 : f32 to f64
    %add = arith.addf %ext, %arg2 : f64
    linalg.yield %add : f64
  } -> tensor<1xf64>
  %init2 = tensor.empty() : tensor<1xf32>
  %result = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%accum : tensor<1xf64>) outs(%init2 : tensor<1xf32>) {
  ^bb0(%arg1: f64, %arg2: f32):
    %res = arith.truncf %arg1 : f64 to f32
    linalg.yield %res : f32
  } -> tensor<1xf32>
  check.expect_almost_eq_const(%result, dense<[25165824.0]> : tensor<1xf32>) : tensor<1xf32>
  return
}

#map = affine_map<() -> ()>
func.func @simpleDAG() {
  %arg0 = arith.constant dense<1.0> : tensor<f32>
  %arg1 = arith.constant dense<2.0> : tensor<f32>
  %arg2 = arith.constant dense<3.0> : tensor<f32>
  %arg3 = arith.constant dense<4.0> : tensor<f32>
  %0 = tensor.empty() : tensor<f32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
      ins(%arg0, %arg1 : tensor<f32>, tensor<f32>) outs(%0 : tensor<f32>) {
    ^bb0(%b0: f32, %b1 : f32, %b2 : f32) :
      %2 = arith.addf %b0, %b1 : f32
      linalg.yield %2 : f32
    } -> tensor<f32>
  %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
      ins(%arg2, %arg3 : tensor<f32>, tensor<f32>) outs(%0 : tensor<f32>) {
    ^bb0(%b0: f32, %b1 : f32, %b2 : f32) :
      %4 = arith.mulf %b0, %b1 : f32
      linalg.yield %4 : f32
    } -> tensor<f32>
  %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
      ins(%1, %3 : tensor<f32>, tensor<f32>) outs(%0 : tensor<f32>) {
    ^bb0(%b0: f32, %b1 : f32, %b2 : f32) :
      %6 = arith.subf %b1, %b0 : f32
      linalg.yield %6 : f32
    } -> tensor<f32>
  check.expect_almost_eq_const(%1, dense<3.0> : tensor<f32>) : tensor<f32>
  check.expect_almost_eq_const(%5, dense<9.0> : tensor<f32>) : tensor<f32>
  return
}

func.func @simpleHorizontal() {
  %arg0 = arith.constant dense<1.0> : tensor<f32>
  %arg1 = arith.constant dense<2.0> : tensor<f32>
  %arg2 = arith.constant dense<3.0> : tensor<f32>
  %arg3 = arith.constant dense<4.0> : tensor<f32>
  %0 = tensor.empty() : tensor<f32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
      ins(%arg0, %arg1 : tensor<f32>, tensor<f32>) outs(%0 : tensor<f32>) {
    ^bb0(%b0: f32, %b1 : f32, %b2 : f32) :
      %2 = arith.addf %b0, %b1 : f32
      linalg.yield %2 : f32
    } -> tensor<f32>
  %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
      ins(%1, %arg2 : tensor<f32>, tensor<f32>) outs(%0 : tensor<f32>) {
    ^bb0(%b0: f32, %b1 : f32, %b2 : f32) :
      %4 = arith.mulf %b0, %b1 : f32
      linalg.yield %4 : f32
    } -> tensor<f32>
  %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []}
      ins(%arg3 : tensor<f32>) outs(%0 : tensor<f32>) {
    ^bb0(%b0: f32, %b1 : f32) :
      %6 = arith.addf %b0, %b0 : f32
      linalg.yield %6 : f32
    } -> tensor<f32>
  check.expect_almost_eq_const(%3, dense<9.0> : tensor<f32>) : tensor<f32>
  check.expect_almost_eq_const(%5, dense<8.0> : tensor<f32>) : tensor<f32>
  return
}

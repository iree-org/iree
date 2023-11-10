#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @softmax(%arg0: tensor<16x128x128xf32>) -> tensor<16x128x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %3 = tensor.empty() : tensor<16x128x128xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %2 : tensor<16x128x128xf32>, tensor<16x128xf32>) outs(%3 : tensor<16x128x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %6 = arith.addf %in, %in_0 : f32
      linalg.yield %6 : f32
    } -> tensor<16x128x128xf32>
    %5 = util.optimization_barrier %4 : tensor<16x128x128xf32>
    return %4 : tensor<16x128x128xf32>
  }
}

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @softmax() -> tensor<16x128x128xf32> {
    %cst = arith.constant -3.40282347E+38 : f32
    %cst_0 = arith.constant dense<5.000000e+00> : tensor<16x128x128xf32>
    %0 = util.optimization_barrier %cst_0 : tensor<16x128x128xf32>
    %1 = tensor.empty() : tensor<16x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cst_0 : tensor<16x128x128xf32>) outs(%2 : tensor<16x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.maxf %in, %out : f32
      linalg.yield %6 : f32
    } -> tensor<16x128xf32>
    %4 = tensor.empty() : tensor<16x128x128xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_0, %3 : tensor<16x128x128xf32>, tensor<16x128xf32>) outs(%4 : tensor<16x128x128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %6 = arith.subf %in, %in_1 : f32
      %7 = math.exp %6 : f32
      linalg.yield %7 : f32
    } -> tensor<16x128x128xf32>
    return %5 : tensor<16x128x128xf32>
  }
}

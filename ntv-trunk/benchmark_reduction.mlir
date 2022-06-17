func.func @reduce() -> (tensor<1x128xf32>) {
  %cst = arith.constant -0.000000e+00 : f32
  
  %arg = util.unfoldable_constant dense<1.0> : tensor<1x128x384xf32>
  %0 = linalg.init_tensor [1, 128] : tensor<1x128xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x128xf32>) ->   tensor<1x128xf32>
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%arg : tensor<1x128x384xf32>) outs(%1 : tensor<1x128xf32>) {
      ^bb0(%arg3: f32, %arg4: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
      } -> tensor<1x128xf32>
  return %2 : tensor<1x128xf32>
}


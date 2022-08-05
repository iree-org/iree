func.func @max_sub_exp() {
  %cst = arith.constant -3.40282347E+38 : f32
  %cst_0 = arith.constant dense<1.000000e+00> : tensor<12x128x128xf32>
  %cst_1 = arith.constant dense<5.000000e+00> : tensor<12x128x128xf32>
  %0 = util.do_not_optimize(%cst_1) : tensor<12x128x128xf32>
  %1 = linalg.init_tensor [12, 128] : tensor<12x128xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<12x128xf32>) -> tensor<12x128xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0 : tensor<12x128x128xf32>) outs(%2 : tensor<12x128xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %8 = arith.cmpf ogt, %arg0, %arg1 : f32
    %9 = arith.select %8, %arg0, %arg1 : f32
    linalg.yield %9 : f32
  } -> tensor<12x128xf32>
  %4 = linalg.init_tensor [12, 128, 128] : tensor<12x128x128xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %3 : tensor<12x128x128xf32>, tensor<12x128xf32>) outs(%4 : tensor<12x128x128xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %8 = arith.subf %arg0, %arg1 : f32
    linalg.yield %8 : f32
  } -> tensor<12x128x128xf32>
  %6 = linalg.init_tensor [12, 128, 128] : tensor<12x128x128xf32>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<12x128x128xf32>) outs(%6 : tensor<12x128x128xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %8 = math.exp %arg0 : f32
    linalg.yield %8 : f32
  } -> tensor<12x128x128xf32>
  check.expect_almost_eq(%7, %cst_0) : tensor<12x128x128xf32>
  return
}


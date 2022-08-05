func.func private @softmax() {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant -3.40282347E+38 : f32
  %cst_2 = arith.constant dense<7.812500e-03> : tensor<12x128x128xf32>
  %cst_3 = arith.constant dense<5.000000e+00> : tensor<12x128x128xf32>
  %0 = util.do_not_optimize(%cst_3) : tensor<12x128x128xf32>
  %1 = linalg.init_tensor [12, 128] : tensor<12x128xf32>
  %2 = linalg.fill ins(%cst_1 : f32) outs(%1 : tensor<12x128xf32>) -> tensor<12x128xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0 : tensor<12x128x128xf32>) outs(%2 : tensor<12x128xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %11 = arith.cmpf ogt, %arg0, %arg1 : f32
    %12 = arith.select %11, %arg0, %arg1 : f32
    linalg.yield %12 : f32
  } -> tensor<12x128xf32>
  %4 = linalg.init_tensor [12, 128, 128] : tensor<12x128x128xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %3 : tensor<12x128x128xf32>, tensor<12x128xf32>) outs(%4 : tensor<12x128x128xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %11 = arith.subf %arg0, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<12x128x128xf32>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<12x128x128xf32>) outs(%4 : tensor<12x128x128xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %11 = math.exp %arg0 : f32
    linalg.yield %11 : f32
  } -> tensor<12x128x128xf32>
  %7 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<12x128xf32>) -> tensor<12x128xf32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6 : tensor<12x128x128xf32>) outs(%7 : tensor<12x128xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %11 = arith.addf %arg0, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<12x128xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<12x128xf32>) outs(%1 : tensor<12x128xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %11 = arith.divf %cst, %arg0 : f32
    linalg.yield %11 : f32
  } -> tensor<12x128xf32>
  %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %9 : tensor<12x128x128xf32>, tensor<12x128xf32>) outs(%4 : tensor<12x128x128xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %11 = arith.mulf %arg0, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<12x128x128xf32>
  check.expect_almost_eq(%10, %cst_2) : tensor<12x128x128xf32>
  return
}

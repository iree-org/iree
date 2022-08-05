// y = gamma * (x-mean(x)) / rsqrt(var(x) + epsilon) + beta
// Setting gamma = 1.0 and beta = 0.0 for simplicity.
func.func private @layernorm() {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x384xf32>
  %cst_2 = arith.constant dense<9.99999996E-13> : tensor<128x1xf32>
  %cst_3 = arith.constant dense<3.840000e+02> : tensor<128x1xf32>
  %cst_4 = arith.constant dense<5.000000e+00> : tensor<128x384xf32>
  %0 = util.do_not_optimize(%cst_4) : tensor<128x384xf32>
  %1 = util.do_not_optimize(%cst_3) : tensor<128x1xf32>
  %2 = linalg.init_tensor [128] : tensor<128xf32>
  %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<128xf32>) -> tensor<128xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%0 : tensor<128x384xf32>) outs(%3 : tensor<128xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %15 = arith.addf %arg0, %arg1 : f32
    linalg.yield %15 : f32
  } -> tensor<128xf32>
  %5 = linalg.init_tensor [128, 1] : tensor<128x1xf32>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<128x1xf32>) outs(%5 : tensor<128x1xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %15 = arith.divf %cst, %arg0 : f32
    linalg.yield %15 : f32
  } -> tensor<128x1xf32>
  %7 = tensor.collapse_shape %6 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
  %8 = linalg.init_tensor [128, 384] : tensor<128x384xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %4, %7 : tensor<128x384xf32>, tensor<128xf32>, tensor<128xf32>) outs(%8 : tensor<128x384xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32):
    %15 = arith.mulf %arg1, %arg2 : f32
    %16 = arith.subf %arg0, %15 : f32
    linalg.yield %16 : f32
  } -> tensor<128x384xf32>
  %10 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<128xf32>) -> tensor<128xf32>
  %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%9 : tensor<128x384xf32>) outs(%10 : tensor<128xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %15 = arith.mulf %arg0, %arg0 : f32
    %16 = arith.addf %15, %arg1 : f32
    linalg.yield %16 : f32
  } -> tensor<128xf32>
  %12 = util.do_not_optimize(%cst_2) : tensor<128x1xf32>
  %13 = tensor.collapse_shape %12 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %11, %7, %13 : tensor<128x384xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%8 : tensor<128x384xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
    %15 = arith.mulf %arg1, %arg2 : f32
    %16 = arith.addf %15, %arg3 : f32
    %17 = math.rsqrt %16 : f32
    %18 = arith.mulf %arg0, %17 : f32
    linalg.yield %18 : f32
  } -> tensor<128x384xf32>
  check.expect_almost_eq(%14, %cst_1) : tensor<128x384xf32>
  return
}
// y = gamma * (x-mean(x)) / rsqrt(var(x) + epsilon) + beta
// Setting gamma = 1.0 and beta = 0.0 for simplicity.
//
// Generated from this TOSA input:
//
// func.func @layernorm() {
//   %x = util.unfoldable_constant dense<5.0> : tensor<128x384xf32>
//   %c384 = util.unfoldable_constant dense<384.0> : tensor<128x1xf32>
//   %sum = tosa.reduce_sum %x {axis = 1 : i64} : (tensor<128x384xf32>) -> tensor<128x1xf32>
//   %r384 = tosa.reciprocal %c384 : (tensor<128x1xf32>) -> tensor<128x1xf32>
//   %mean = tosa.mul %sum, %r384 {shift = 0 : i8} : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
//   %x_sub_mean = tosa.sub %x, %mean : (tensor<128x384xf32>, tensor<128x1xf32>) -> tensor<128x384xf32>
//   %square = tosa.mul %x_sub_mean, %x_sub_mean {shift = 0 : i8} : (tensor<128x384xf32>, tensor<128x384xf32>) -> tensor<128x384xf32>
//   %square_sum = tosa.reduce_sum %square {axis = 1 : i64} : (tensor<128x384xf32>) -> tensor<128x1xf32>
//   %variance = tosa.mul %square_sum, %r384 {shift = 0 : i8} : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
//   %epsilon = util.unfoldable_constant dense<9.99999996E-13> : tensor<128x1xf32>
//   %var_eps = tosa.add %variance, %epsilon : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
//   %rsigma = tosa.rsqrt %var_eps : (tensor<128x1xf32>) -> tensor<128x1xf32>
//   %norm = tosa.mul %x_sub_mean, %rsigma {shift = 0 : i8} : (tensor<128x384xf32>, tensor<128x1xf32>) -> tensor<128x384xf32>
//   check.expect_almost_eq_const(%norm, dense<0.0> : tensor<128x384xf32>) : tensor<128x384xf32>
//   return
// }

func.func @layernorm() {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x384xf32>
  %cst_2 = arith.constant dense<9.99999996E-13> : tensor<128x1xf32>
  %cst_3 = arith.constant dense<3.840000e+02> : tensor<128x1xf32>
  %cst_4 = arith.constant dense<5.000000e+00> : tensor<128x384xf32>
  %0 = util.optimization_barrier %cst_4 : tensor<128x384xf32>
  %1 = util.optimization_barrier %cst_3 : tensor<128x1xf32>
  %2 = tensor.empty() : tensor<128xf32>
  %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<128xf32>) -> tensor<128xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%0 : tensor<128x384xf32>) outs(%3 : tensor<128xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %15 = arith.addf %arg0, %arg1 : f32
    linalg.yield %15 : f32
  } -> tensor<128xf32>
  %5 = tensor.empty() : tensor<128x1xf32>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<128x1xf32>) outs(%5 : tensor<128x1xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %15 = arith.divf %cst, %arg0 : f32
    linalg.yield %15 : f32
  } -> tensor<128x1xf32>
  %7 = tensor.collapse_shape %6 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
  %8 = tensor.empty() : tensor<128x384xf32>
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
  %12 = util.optimization_barrier %cst_2 : tensor<128x1xf32>
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

func.func @layernorm_dynamic() {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x384xf32>
  %cst_2 = flow.tensor.constant dense<9.99999996E-13> : tensor<128x1xf32> -> tensor<?x1xf32>
  %cst_3 = flow.tensor.constant dense<3.840000e+02> : tensor<128x1xf32> -> tensor<?x1xf32>
  %cst_4 = flow.tensor.constant dense<5.000000e+00> : tensor<128x384xf32> -> tensor<?x?xf32>
  %c_0_index = arith.constant 0 : index
  %c_1_index = arith.constant 1 : index
  %dim_0 = tensor.dim %cst_4, %c_0_index : tensor<?x?xf32>
  %dim_1 = tensor.dim %cst_4, %c_1_index : tensor<?x?xf32>
  %2 = tensor.empty(%dim_0) : tensor<?xf32>
  %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<?xf32>) -> tensor<?xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%cst_4 : tensor<?x?xf32>) outs(%3 : tensor<?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %15 = arith.addf %arg0, %arg1 : f32
    linalg.yield %15 : f32
  } -> tensor<?xf32>
  %5 = tensor.empty(%dim_0) : tensor<?x1xf32>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cst_3 : tensor<?x1xf32>) outs(%5 : tensor<?x1xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %15 = arith.divf %cst, %arg0 : f32
    linalg.yield %15 : f32
  } -> tensor<?x1xf32>
  %7 = tensor.collapse_shape %6 [[0, 1]] : tensor<?x1xf32> into tensor<?xf32>
  %8 = tensor.empty(%dim_0, %dim_1) : tensor<?x?xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cst_4, %4, %7 : tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%8 : tensor<?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32):
    %15 = arith.mulf %arg1, %arg2 : f32
    %16 = arith.subf %arg0, %15 : f32
    linalg.yield %16 : f32
  } -> tensor<?x?xf32>
  %10 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<?xf32>) -> tensor<?xf32>
  %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%9 : tensor<?x?xf32>) outs(%10 : tensor<?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %15 = arith.mulf %arg0, %arg0 : f32
    %16 = arith.addf %15, %arg1 : f32
    linalg.yield %16 : f32
  } -> tensor<?xf32>
  %13 = tensor.collapse_shape %cst_2 [[0, 1]] : tensor<?x1xf32> into tensor<?xf32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %11, %7, %13 : tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%8 : tensor<?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
    %15 = arith.mulf %arg1, %arg2 : f32
    %16 = arith.addf %15, %arg3 : f32
    %17 = math.rsqrt %16 : f32
    %18 = arith.mulf %arg0, %17 : f32
    linalg.yield %18 : f32
  } -> tensor<?x?xf32>
  %result = tensor.cast %14 : tensor<?x?xf32> to tensor<128x384xf32>
  check.expect_almost_eq(%result, %cst_1) : tensor<128x384xf32>
  return
}

// Generated from this TOSA input:
//
// func.func @softmax() {
//   %0 = util.unfoldable_constant dense<5.0> : tensor<12x128x128xf32>
//   %red = tosa.reduce_max %0 {axis = 2 : i64} : (tensor<12x128x128xf32>) -> tensor<12x128x1xf32>
//   %sub = tosa.sub %0, %red : (tensor<12x128x128xf32>, tensor<12x128x1xf32>) -> tensor<12x128x128xf32>
//   %exp = tosa.exp %sub : (tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
//   %sum = tosa.reduce_sum %exp {axis = 2 : i64} : (tensor<12x128x128xf32>) -> tensor<12x128x1xf32>
//   %rec = tosa.reciprocal %sum : (tensor<12x128x1xf32>) -> tensor<12x128x1xf32>
//   %mul = tosa.mul %exp, %rec {shift = 0 : i8} : (tensor<12x128x128xf32>, tensor<12x128x1xf32>) -> tensor<12x128x128xf32>
//   check.expect_almost_eq_const(%mul, dense<0.0078125> : tensor<12x128x128xf32>) : tensor<12x128x128xf32>
//   return
// }

func.func @softmax() {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant -3.40282347E+38 : f32
  %cst_2 = arith.constant dense<7.812500e-03> : tensor<12x128x128xf32>
  %cst_3 = arith.constant dense<5.000000e+00> : tensor<12x128x128xf32>
  %0 = util.optimization_barrier %cst_3 : tensor<12x128x128xf32>
  %1 = tensor.empty() : tensor<12x128xf32>
  %2 = linalg.fill ins(%cst_1 : f32) outs(%1 : tensor<12x128xf32>) -> tensor<12x128xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0 : tensor<12x128x128xf32>) outs(%2 : tensor<12x128xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %11 = arith.maximumf %arg0, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<12x128xf32>
  %4 = tensor.empty() : tensor<12x128x128xf32>
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

func.func @softmax_dynamic() {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant -3.40282347E+38 : f32
  %cst_2 = arith.constant dense<7.812500e-03> : tensor<12x128x128xf32>
  %cst_3 = flow.tensor.constant dense<5.000000e+00> : tensor<12x128x128xf32> -> tensor<?x?x?xf32>
  %c_0_index = arith.constant 0 : index
  %c_1_index = arith.constant 1 : index
  %c_2_index = arith.constant 2 : index
  %dim_0 = tensor.dim %cst_3, %c_0_index : tensor<?x?x?xf32>
  %dim_1 = tensor.dim %cst_3, %c_1_index : tensor<?x?x?xf32>
  %dim_2 = tensor.dim %cst_3, %c_2_index : tensor<?x?x?xf32>
  %1 = tensor.empty(%dim_0, %dim_1) : tensor<?x?xf32>
  %2 = linalg.fill ins(%cst_1 : f32) outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cst_3 : tensor<?x?x?xf32>) outs(%2 : tensor<?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %11 = arith.maximumf %arg0, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?xf32>
  %4 = tensor.empty(%dim_0, %dim_1, %dim_2) : tensor<?x?x?xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_3, %3 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %11 = arith.subf %arg0, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?x?xf32>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<?x?x?xf32>) outs(%4 : tensor<?x?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %11 = math.exp %arg0 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?x?xf32>
  %7 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6 : tensor<?x?x?xf32>) outs(%7 : tensor<?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %11 = arith.addf %arg0, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<?x?xf32>) outs(%1 : tensor<?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %11 = arith.divf %cst, %arg0 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?xf32>
  %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %9 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %11 = arith.mulf %arg0, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?x?xf32>
  %result = tensor.cast %10 : tensor<?x?x?xf32> to tensor<12x128x128xf32>
  check.expect_almost_eq(%result, %cst_2) : tensor<12x128x128xf32>
  return
}

// RUN: iree-opt --iree-flow-raise-special-ops -canonicalize %s | FileCheck %s

// CHECK-LABEL: @softmax
//  CHECK-SAME: %[[ARG:.+]]: tensor<?x?x?xf32>
//       CHECK:   %[[E:.+]] = tensor.empty(%{{.*}}, %{{.*}}, %{{.*}}) : tensor<?x?x?xf32>
//       CHECK:   %[[S:.+]] = iree_linalg_ext.softmax dimension(2) ins(%[[ARG]] : tensor<?x?x?xf32>) outs(%[[E]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
//       CHECK:   return %[[S]] : tensor<?x?x?xf32>

func.func @softmax(%src : tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>) {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant -3.40282347E+38 : f32
  %c_0_index = arith.constant 0 : index
  %c_1_index = arith.constant 1 : index
  %c_2_index = arith.constant 2 : index
  %dim_0 = tensor.dim %src, %c_0_index : tensor<?x?x?xf32>
  %dim_1 = tensor.dim %src, %c_1_index : tensor<?x?x?xf32>
  %dim_2 = tensor.dim %src, %c_2_index : tensor<?x?x?xf32>
  %1 = tensor.empty(%dim_0, %dim_1) : tensor<?x?xf32>
  %2 = linalg.fill ins(%cst_1 : f32) outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%src : tensor<?x?x?xf32>) outs(%2 : tensor<?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %11 = arith.maxf %arg0, %arg1 : f32
    linalg.yield %11 : f32
  } -> tensor<?x?xf32>
  %4 = tensor.empty(%dim_0, %dim_1, %dim_2) : tensor<?x?x?xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%src, %3 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?x?xf32>) {
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
  return %10 : tensor<?x?x?xf32>
}

// CHECK-LABEL: @softmax_no_rcp
//  CHECK-SAME: %[[ARG:.+]]: tensor<10x4096x4096xf16>
//       CHECK:   %[[E:.+]] = tensor.empty() : tensor<10x4096x4096xf16>
//       CHECK:   %[[S:.+]] = iree_linalg_ext.softmax dimension(2) ins(%[[ARG]] : tensor<10x4096x4096xf16>) outs(%[[E]] : tensor<10x4096x4096xf16>) -> tensor<10x4096x4096xf16>
//       CHECK:   return %[[S]] : tensor<10x4096x4096xf16>
func.func @softmax_no_rcp(%src : tensor<10x4096x4096xf16>) -> (tensor<10x4096x4096xf16>) {
  %cst_158 = arith.constant -6.550400e+04 : f16
  %cst_121 = arith.constant 0.000000e+00 : f16
  %224 = tensor.empty() : tensor<10x4096xf16>
  %216 = tensor.empty() : tensor<10x4096x4096xf16>
  %225 = linalg.fill ins(%cst_158 : f16) outs(%224 : tensor<10x4096xf16>) -> tensor<10x4096xf16>
  %226 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%src : tensor<10x4096x4096xf16>) outs(%225 : tensor<10x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5290 = arith.maxf %in, %out : f16
    linalg.yield %5290 : f16
  } -> tensor<10x4096xf16>
  %227 = linalg.generic
  {indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    affine_map<(d0, d1, d2) -> (d0, d1)>,
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%src, %226 : tensor<10x4096x4096xf16>, tensor<10x4096xf16>) outs(%216 : tensor<10x4096x4096xf16>) {
  ^bb0(%in: f16, %in_1572: f16, %out: f16):
    %5290 = arith.subf %in, %in_1572 : f16
    linalg.yield %5290 : f16
  } -> tensor<10x4096x4096xf16>
  %228 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%227 : tensor<10x4096x4096xf16>) outs(%216 : tensor<10x4096x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5290 = math.exp %in : f16
    linalg.yield %5290 : f16
  } -> tensor<10x4096x4096xf16>
  %229 = tensor.empty() : tensor<10x4096xf16>
  %230 = linalg.fill ins(%cst_121 : f16) outs(%229 : tensor<10x4096xf16>) -> tensor<10x4096xf16>
  %231 = linalg.generic 
  {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel", "reduction"]}
  ins(%228 : tensor<10x4096x4096xf16>) outs(%230 : tensor<10x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %5290 = arith.addf %in, %out : f16
    linalg.yield %5290 : f16
  } -> tensor<10x4096xf16>
  %232 = linalg.generic 
  {indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    affine_map<(d0, d1, d2) -> (d0, d1)>,
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%228, %231 : tensor<10x4096x4096xf16>, tensor<10x4096xf16>) outs(%216 : tensor<10x4096x4096xf16>) {
  ^bb0(%in: f16, %in_1572: f16, %out: f16):
    %5290 = arith.divf %in, %in_1572 : f16
    linalg.yield %5290 : f16
  } -> tensor<10x4096x4096xf16>
  return %232 : tensor<10x4096x4096xf16>
}

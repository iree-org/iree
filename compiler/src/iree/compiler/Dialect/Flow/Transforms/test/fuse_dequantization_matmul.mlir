// RUN: iree-opt --split-input-file --iree-flow-fuse-dequantization-matmul --canonicalize %s | FileCheck %s

module {
  func.func @clone_grouped_quantized_matmul(%arg0: tensor<4096x32x128xi8>, %arg1: tensor<1x1x32x128xf32>, %arg2: tensor<4096x32x1xf32>, %arg3: tensor<4096x32x1xf32>) -> tensor<1x1x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x4096xf32>
    %1 = tensor.empty() : tensor<4096x32x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
    %3 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1, 0)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1, 0)>, 
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>], 
        iterator_types = ["parallel", "parallel", "parallel"]} 
        ins(%arg0, %arg2, %arg3 : tensor<4096x32x128xi8>, tensor<4096x32x1xf32>, tensor<4096x32x1xf32>) outs(%1 : tensor<4096x32x128xf32>) {
    ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
      %5 = arith.extui %in : i8 to i32
      %6 = arith.uitofp %5 : i32 to f32
      %7 = arith.subf %6, %in_1 : f32
      %8 = arith.mulf %7, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<4096x32x128xf32>
    %barrier = util.optimization_barrier %3 : tensor<4096x32x128xf32>
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, 
                         affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>, 
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], 
        iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} 
        ins(%arg1, %3 : tensor<1x1x32x128xf32>, tensor<4096x32x128xf32>) outs(%2 : tensor<1x1x4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<1x1x4096xf32>
    return %4 : tensor<1x1x4096xf32>
  }
}
//       CHECK: func.func @clone_grouped_quantized_matmul
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<4096x32x128xi8>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1x1x32x128xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<4096x32x1xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<4096x32x1xf32>
//       CHECK:   %[[C0:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[INIT1:.+]] = tensor.empty() : tensor<1x1x4096xf32>
//       CHECK:   %[[INIT0:.+]] = tensor.empty() : tensor<4096x32x128xf32>
//       CHECK:   %[[GEN0:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG2]], %[[ARG3]] :
//  CHECK-SAME:       outs(%[[INIT0]] :
//       CHECK:   %[[DISP:.+]] = flow.dispatch.region -> (tensor<1x1x4096xf32>)
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[C0]]
//  CHECK-SAME:       outs(%[[INIT1]] :
//       CHECK:   %[[CLONE:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG2]], %[[ARG3]] :
//  CHECK-SAME:       outs(%[[INIT0]] :
//       CHECK:   %[[GEN1:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
//  CHECK-SAME:       ins(%[[ARG1]], %[[CLONE]] :
//  CHECK-SAME:       outs(%[[FILL]] :
//       CHECK:   flow.return %[[GEN1]] :
//       CHECK:   return %[[DISP]]
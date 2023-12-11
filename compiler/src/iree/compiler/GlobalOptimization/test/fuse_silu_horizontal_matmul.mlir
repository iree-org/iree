// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-global-opt-fuse-silu-horizontal-matmul,canonicalize))" %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @silu_horizontal_matmul_fusion(%arg0: index, %arg1: tensor<?x5120xf16>, %arg2: tensor<13824x5120xf16>, %arg3: tensor<13824x5120xf16>) -> tensor<?x13824xf16> {
    %cst = arith.constant 1.000000e+00 : f16
    %cst_0 = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty(%arg0) : tensor<?x13824xf16>
    %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<?x13824xf16>) -> tensor<?x13824xf16>
    %2 = linalg.matmul_transpose_b ins(%arg1, %arg2 : tensor<?x5120xf16>, tensor<13824x5120xf16>) outs(%1 : tensor<?x13824xf16>) -> tensor<?x13824xf16>
    %3 = tensor.empty(%arg0) : tensor<?x13824xf16>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<?x13824xf16>) outs(%3 : tensor<?x13824xf16>) {
    ^bb0(%in: f16, %out: f16):
      %10 = arith.negf %in : f16
      %11 = math.exp %10 : f16
      %12 = arith.addf %11, %cst_0 : f16
      %13 = arith.divf %cst_0, %12 : f16
      linalg.yield %13 : f16
    } -> tensor<?x13824xf16>
    %5 = tensor.empty(%arg0) : tensor<?x13824xf16>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %2 : tensor<?x13824xf16>, tensor<?x13824xf16>) outs(%5 : tensor<?x13824xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %10 = arith.mulf %in, %in_1 : f16
      linalg.yield %10 : f16
    } -> tensor<?x13824xf16>
    %7 = linalg.matmul_transpose_b ins(%arg1, %arg3 : tensor<?x5120xf16>, tensor<13824x5120xf16>) outs(%1 : tensor<?x13824xf16>) -> tensor<?x13824xf16>
    %8 = tensor.empty(%arg0) : tensor<?x13824xf16>
    %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%6, %7 : tensor<?x13824xf16>, tensor<?x13824xf16>) outs(%8 : tensor<?x13824xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %10 = arith.mulf %in, %in_1 : f16
      linalg.yield %10 : f16
    } -> tensor<?x13824xf16>
    return %9 : tensor<?x13824xf16>
  }
}
//   CHECK-DAG: #[[MAP:[a-zA-Z0-9]+]] = affine_map<(d0, d1) -> (d0, d1)>
//       CHECK: func.func @silu_horizontal_matmul_fusion(
//  CHECK-SAME:   %[[IN0:.+]]: index,
//  CHECK-SAME:   %[[IN1:.+]]: tensor<?x5120xf16>,
//  CHECK-SAME:   %[[IN2:.+]]: tensor<13824x5120xf16>,
//  CHECK-SAME:   %[[IN3:.+]]: tensor<13824x5120xf16>)
//  CHECK-SAME:   -> tensor<?x13824xf16> {
//       CHECK:   %[[CST:.+]] = arith.constant 1.000000e+00 : f16
//       CHECK:   %[[CST_0:.+]] = arith.constant 0.000000e+00 : f16
//       CHECK:   %[[INIT0:.+]] = tensor.empty(%[[IN0]]) : tensor<?x13824xf16>
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f16) outs(%[[INIT0]] : tensor<?x13824xf16>) -> tensor<?x13824xf16>
//       CHECK:   %[[INIT1:.+]] = tensor.empty(%[[IN0]]) : tensor<?x13824xf16>
//       CHECK:   %[[INIT2:.+]] = tensor.empty(%[[IN0]]) : tensor<?x13824xf16>
//       CHECK:   %[[INIT3:.+]] = tensor.empty(%[[IN0]]) : tensor<?x13824xf16>
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region -> (tensor<?x13824xf16>{%[[IN0:.+]]}) {
//       CHECK:   %[[MATMUL1:.+]] = linalg.matmul_transpose_b ins(%[[IN1]], %[[IN2]] : tensor<?x5120xf16>, tensor<13824x5120xf16>) outs(%[[FILL]] : tensor<?x13824xf16>) -> tensor<?x13824xf16>
//       CHECK:   %[[SIGMOID:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel"]} ins(%[[MATMUL1]] : tensor<?x13824xf16>) outs(%[[INIT1]] : tensor<?x13824xf16>) {
//       CHECK:   ^bb0(%[[SIGMOIDINPUT:.+]]: f16, %out: f16):
//       CHECK:     %[[SIGMOID1:.+]]= arith.negf %[[SIGMOIDINPUT]] : f16
//       CHECK:     %[[SIGMOID2:.+]] = math.exp %[[SIGMOID1]]: f16
//       CHECK:     %[[SIGMOID3:.+]] = arith.addf %[[SIGMOID2]], %[[CST_0]] : f16
//       CHECK:     %[[SIGMOID4:.+]] = arith.divf %[[CST_0]], %[[SIGMOID3]] : f16
//       CHECK:     linalg.yield %[[SIGMOID4]] : f16
//       CHECK:   } -> tensor<?x13824xf16>
//       CHECK:   %[[SILU:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel"]} ins(%[[SIGMOID]], %[[MATMUL1]] : tensor<?x13824xf16>, tensor<?x13824xf16>) outs(%[[INIT2]] : tensor<?x13824xf16>) {
//       CHECK:   ^bb0(%[[SILUINPUT1:.+]]: f16, %[[SILUINPUT2:.+]]: f16, %out: f16):
//       CHECK:     %[[SILU5:.+]]= arith.mulf %[[SILUINPUT1:.+]], %[[SILUINPUT2:.+]] : f16
//       CHECK:     linalg.yield %[[SILU5:.+]]: f16
//       CHECK:   } -> tensor<?x13824xf16>
//       CHECK:   %[[MATMUL2:.+]] = linalg.matmul_transpose_b ins(%[[IN1]], %[[IN3]] : tensor<?x5120xf16>, tensor<13824x5120xf16>) outs(%[[FILL]] : tensor<?x13824xf16>) -> tensor<?x13824xf16>
//       CHECK:   %[[OUTPUT:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel"]} ins(%[[SILU]], %[[MATMUL2]] : tensor<?x13824xf16>, tensor<?x13824xf16>) outs(%[[INIT3]] : tensor<?x13824xf16>) {
//       CHECK:   ^bb0(%[[MULTIN0:.+]]: f16, %[[MULTIN1:.+]]: f16, %out: f16):
//       CHECK:     %[[MULT:.+]]= arith.mulf %[[MULTIN0]], %[[MULTIN1]] : f16
//       CHECK:     linalg.yield %[[MULT]]: f16
//       CHECK:   } -> tensor<?x13824xf16>
//       CHECK:   flow.return %[[OUTPUT]] : tensor<?x13824xf16>
//       CHECK: }
//       CHECK: return %[[DISPATCH]] : tensor<?x13824xf16>

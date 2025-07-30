// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-fuse-multi-use-elementwise-producer{intra-dispatch}))" --split-input-file %s | FileCheck %s

util.func public @multi_trunc_consumers(%arg0: tensor<4x2048xi64>, %arg1: tensor<4x2048xi64>, %arg2: tensor<4x2048xi32>) -> (tensor<4x2048xi32>, tensor<4x2048xi32>) {
  %c36_i64 = arith.constant 36 : i64
  %c2_i64 = arith.constant 2 : i64
  %c1_i64 = arith.constant 1 : i64
  %dispatch:2 = flow.dispatch.region -> (tensor<4x2048xi32>, tensor<4x2048xi32>){
    %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<4x2048xi64>) outs(
  %arg1 : tensor<4x2048xi64>) {
    ^bb0(%in: i64, %out: i64):
      %3 = arith.addi %in, %c36_i64 : i64
      %4 = arith.muli %3, %c2_i64 : i64
      linalg.yield %4 : i64
    } -> tensor<4x2048xi64>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<4x2048xi64>) outs(%arg2 : tensor<4x2048xi32>) {
    ^bb0(%in: i64, %out: i32):
      %3 = arith.trunci %in : i64 to i32
      linalg.yield %3 : i32
    } -> tensor<4x2048xi32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<4x2048xi64>) outs(%arg2 : tensor<4x2048xi32>) {
    ^bb0(%in: i64, %out: i32):
      %3 = arith.addi %in, %c1_i64 : i64
      %4 = arith.trunci %3 : i64 to i32
      linalg.yield %4 : i32
    } -> tensor<4x2048xi32>
    flow.return %2, %1 : tensor<4x2048xi32>, tensor<4x2048xi32>
  }
  util.return %dispatch#0, %dispatch#1 : tensor<4x2048xi32>, tensor<4x2048xi32>
}

// CHECK-LABEL: util.func public @multi_trunc_consumers(
//       CHECK:   %[[GENERIC:.+]]:2 = linalg.generic
//       CHECK:   flow.return %[[GENERIC]]#1, %[[GENERIC]]#0

// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-aggregated-ops), canonicalize, cse)" --split-input-file %s | FileCheck %s

func.func @attention(
  %s_in: tensor<20x4096x4096xf32>,
  %v_in: tensor<20x4096x64xf32>,
  %max_init: tensor<20x4096xf32>,
  %sum_init: tensor<20x4096xf32>,
  %acc_init: tensor<20x4096x64xf32>
) -> (tensor<20x4096xf32>, tensor<20x4096xf32>, tensor<20x4096x64xf32>)
  {
  %max, %sumt, %pv = iree_linalg_ext.exp_reduction {
    indexing_maps = [
      affine_map<(B, M, N, K2) -> (B, M, K2)>,
      affine_map<(B, M, N, K2) -> (B, K2, N)>,
      affine_map<(B, M, N, K2) -> (B, M)>,
      affine_map<(B, M, N, K2) -> (B, M)>,
      affine_map<(B, M, N, K2) -> (B, M, N)>
    ],
    iterator_types = [
      #iree_linalg_ext.iterator_type<parallel>,
      #iree_linalg_ext.iterator_type<parallel>,
      #iree_linalg_ext.iterator_type<parallel>,
      #iree_linalg_ext.iterator_type<reduction>
    ],
    exp_reduced_operands = [1, 2]
  }
    ins(%s_in, %v_in : tensor<20x4096x4096xf32>, tensor<20x4096x64xf32>)
    outs(%max_init, %sum_init, %acc_init : tensor<20x4096xf32>, tensor<20x4096xf32>, tensor<20x4096x64xf32>)
  {
  ^bb0(%ex : f32, %v : f32, %m : f32, %sum : f32, %acc : f32):
    %nsum = arith.addf %ex, %sum : f32
    %mul  = arith.mulf %ex, %v : f32
    %nacc = arith.addf %mul, %acc : f32
    iree_linalg_ext.yield %m, %nsum, %nacc : f32, f32, f32
  } -> tensor<20x4096xf32>, tensor<20x4096xf32>, tensor<20x4096x64xf32>

  return %max, %sumt, %pv : tensor<20x4096xf32>, tensor<20x4096xf32>, tensor<20x4096x64xf32>
}
// CHECK-LABEL: @attention
// CHECK-SAME: %[[S:[0-9A-Za-z]*]]: tensor<20x4096x4096xf32>
// CHECK-SAME: %[[V:[0-9A-Za-z]*]]: tensor<20x4096x64xf32>
// CHECK: %[[M:.*]] = linalg.generic
// CHECK-SAME: ins(%[[S]]
// CHECK-SAME: outs(
// CHECK:   arith.maximumf
// CHECK: %[[sub:.*]] = linalg.generic
// CHECK-SAME: ins(%[[M]]
// CHECK-SAME: outs(%[[S]]
// CHECK:   arith.subf
// CHECK:   math.exp2
// CHECK: %[[n:.*]] = linalg.generic
// CHECK-SAME: ins(%[[M]]
// CHECK:   arith.subf
// CHECK:   math.exp2
// norms
// CHECK: %[[max_norm:.*]] = linalg.generic
// CHECK-SAME: ins(%[[n]]
// CHECK:   arith.mulf
// CHECK: %[[acc_norm:.*]] = linalg.generic
// CHECK-SAME: ins(%[[n]]
// CHECK:   arith.mulf
// reduction body
// CHECK: %[[SUM:.*]] = linalg.generic
// CHECK-SAME: ins(%[[sub]]
// CHECK-SAME: outs(%[[max_norm]]
// CHECK:   arith.addf
// CHECK: %[[PV:.*]] = linalg.generic
// CHECK-SAME: ins(%[[sub]], %[[V]]
// CHECK-SAME: outs(%[[acc_norm]]
// CHECK:   arith.mulf
// CHECK:   arith.addf
// CHECK: return 
// CHECK-SAME: %[[M]]
// CHECK-SAME: %[[SUM]]
// CHECK-SAME: %[[PV]]
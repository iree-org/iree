// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-exp-reduction,cse))" --split-input-file %s | FileCheck %s

builtin.module {
  func.func @attention(
    %S: tensor<20x4096x4096xf32>,
    %V: tensor<20x4096x64xf32>
  ) -> tensor<20x4096x64xf32> 
  {

  %red_empty = tensor.empty() : tensor<20x4096x64xf32>
  %max_empty = tensor.empty() : tensor<20x4096xf32>

  %max_el = arith.constant -3.40282347E+38 : f32
  %max_init = linalg.fill ins(%max_el : f32)
                          outs(%max_empty : tensor<20x4096xf32>)
                          -> tensor<20x4096xf32>

  %sum_empty = tensor.empty() : tensor<20x4096xf32>
  %sum_el = arith.constant 0.000000e+00 : f32
  %sum_init = linalg.fill ins(%sum_el : f32)
                          outs(%sum_empty : tensor<20x4096xf32>)
                          -> tensor<20x4096xf32>
  %acc_init = linalg.fill ins(%sum_el : f32)
                          outs(%red_empty : tensor<20x4096x64xf32>)
                          -> tensor<20x4096x64xf32>


  %MAX, %SUM, %PV = iree_linalg_ext.exp_reduction {
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
    ins(%S, %V : tensor<20x4096x4096xf32>, tensor<20x4096x64xf32>)
    outs(%max_init, %sum_init, %acc_init : tensor<20x4096xf32>, tensor<20x4096xf32>, tensor<20x4096x64xf32>)
  {
  ^bb0(%ex : f32, %v : f32, %m : f32, %sum : f32, %acc : f32):
    %nsum = arith.addf %ex, %sum : f32
    %mul  = arith.mulf %ex, %v : f32
    %nacc = arith.addf %mul, %acc : f32
    iree_linalg_ext.yield %m, %nsum, %nacc : f32, f32, f32
  } -> tensor<20x4096xf32>, tensor<20x4096xf32>, tensor<20x4096x64xf32>

  return %PV : tensor<20x4096x64xf32>
}
}

// CHECK-LABEL: @attention
// CHECK-SAME: %[[S:[a-zA-Z0-9_]+]]: tensor<20x4096x4096xf32>, %[[V:[a-zA-Z0-9_]+]]: tensor<20x4096x64xf32>
// CHECK-DAG: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[NEG_INF:.+]] = arith.constant -3.40282347E+38 : f32
// CHECK-DAG: %[[MAX_FILL:.+]] = linalg.fill ins(%[[NEG_INF]]
// CHECK-DAG: %[[SUM_FILL:.+]] = linalg.fill ins(%[[ZERO]]
// CHECK: %[[MAX:.+]] = linalg.generic
// CHECK:    arith.maximumf
// CHECK: %[[NORM:.+]] = linalg.generic
// CHECK-SAME:     ins(%[[MAX]]
// CHECK-SAME:     outs(%[[S]]
// CHECK:        arith.subf
// CHECK:        math.exp2
// CHECK: %[[ALPHA:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[MAX]]
// CHECK-SAME:    outs(%[[MAX_FILL]]
// CHECK:        arith.subf
// CHECK:        math.exp2
// CHECK: %[[NORM_MUL:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[ALPHA]]
// CHECK-SAME:    outs(%[[SUM_FILL]]
// CHECK:        arith.mulf
// CHECK: %[[EXP:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[NORM]], %[[V]]
// CHECK-SAME:    outs(%[[NORM_MUL]]
// CHECK:        arith.mulf
// CHECK:        arith.addf
// CHECK:        linalg.yield
// CHECK-DAG: return %[[EXP]] : tensor<20x4096x64xf32>
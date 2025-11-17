// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-exp-reduction,cse))" --split-input-file %s | FileCheck %s

builtin.module {
  func.func @attention(
    %S: tensor<20x4096x4096xf32>,
    %V: tensor<20x4096x64xf32>,
    %max_init: tensor<20x4096xf32>,
    %sum_init: tensor<20x4096xf32>,
    %acc_init: tensor<20x4096x64xf32>
  ) -> tensor<20x4096x64xf32> 
  {
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
// CHECK-SAME:     %[[S:[a-zA-Z0-9_]+]]: tensor<20x4096x4096xf32>
// CHECK-SAME:     %[[V:[a-zA-Z0-9_]+]]: tensor<20x4096x64xf32>
// CHECK-SAME:     %[[MAX_FILL:[a-zA-Z0-9_]+]]: tensor<20x4096xf32>
// CHECK-SAME:     %[[SUM_FILL:[a-zA-Z0-9_]+]]: tensor<20x4096xf32>
// CHECK-SAME:     %[[ACC_FILL:[a-zA-Z0-9_]+]]: tensor<20x4096x64xf32>
// CHECK: %[[MAX:.+]] = linalg.generic
// CHECK-SAME:     ins(%[[S]]
// CHECK-SAME:     outs(%[[MAX_FILL]]
// CHECK:        arith.maximumf
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
// CHECK-SAME:    outs(%[[ACC_FILL]]
// CHECK:        arith.mulf
// CHECK: %[[EXP:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[NORM]], %[[V]]
// CHECK-SAME:    outs(%[[NORM_MUL]]
// CHECK:        arith.mulf
// CHECK:        arith.addf
// CHECK:        linalg.yield
// CHECK-DAG: return %[[EXP]] : tensor<20x4096x64xf32>

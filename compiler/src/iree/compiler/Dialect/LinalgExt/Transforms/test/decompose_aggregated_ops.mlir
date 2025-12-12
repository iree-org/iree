// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-aggregated-ops{filter-ops=iree_linalg_ext.online_attention}), canonicalize, cse)" --split-input-file %s | FileCheck %s

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

func.func @online_attention_f8(%query: tensor<192x1024x64xf8E4M3FNUZ>,
                         %key: tensor<192x1024x64xf8E4M3FNUZ>,
                         %value: tensor<192x1024x64xf8E4M3FNUZ>,
                         %output: tensor<192x1024x64xf32>,
                         %max: tensor<192x1024xf32>,
                         %sum: tensor<192x1024xf32>)
                         -> (tensor<192x1024x64xf32>, tensor<192x1024xf32>) {
  %scale = arith.constant 1.0 : f32

  %out:3 = iree_linalg_ext.online_attention
        { indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO, #mapR, #mapR] }
        ins(%query, %key, %value, %scale : tensor<192x1024x64xf8E4M3FNUZ>, tensor<192x1024x64xf8E4M3FNUZ>, tensor<192x1024x64xf8E4M3FNUZ>, f32)
        outs(%output, %max, %sum : tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     }
        -> tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>

  return %out#0, %out#2 : tensor<192x1024x64xf32>, tensor<192x1024xf32>
}
// CHECK-LABEL: @online_attention_f8
// S = Q @ K
// CHECK: linalg.generic
// CHECK:   arith.extf %[[A:.+]] : f8E4M3FNUZ to f32
// CHECK:   arith.extf %[[A:.+]] : f8E4M3FNUZ to f32
// CHECK:   arith.mulf
// CHECK:   arith.addf
// CHECK:   linalg.yield
// S = S * scale
// CHECK:   linalg.generic
// CHECK-NOT: arith.extf
// CHECK:   arith.mulf
// CHECK-NEXT:   linalg.yield
// S = S + F8_linear_offset
// CHECK:   linalg.generic
// CHECK-NOT: arith.extf
// CHECK:   arith.addf
// CHECK-NEXT:   linalg.yield
// newMax = max(oldMax, rowMax(S))
// CHECK: linalg.generic
// CHECK-NOT: arith.extf
// CHECK:   arith.maximumf
// CHECK:   linalg.yield
// norm = exp2(oldMax - newMax)
// CHECK: linalg.generic
// CHECK-NOT: arith.extf
// CHECK:   arith.subf
// CHECK:   math.exp2
// CHECK:   linalg.yield
// normSum = norm * oldSum
// CHECK: linalg.generic
// CHECK-NOT: arith.extf
// CHECK:   arith.mulf
// CHECK:   linalg.yield
// P = exp2(S - newMax)
// CHECK: linalg.generic
// CHECK-NOT: arith.extf
// CHECK:   arith.subf
// CHECK:   math.exp2
// CHECK:   linalg.yield
// newSum = normSum + rowSum(P)
// CHECK: linalg.generic
// CHECK-NOT: arith.extf
// CHECK:   arith.addf
// CHECK:   linalg.yield
// clamp = clamp(norm)
// CHECK: linalg.generic
// CHECK-NOT: arith.extf
// CHECK:   arith.minimumf
// CHECK:   arith.truncf
// newAcc = norm * oldAcc
// CHECK: linalg.generic
// CHECK-NOT: arith.extf
// CHECK:   arith.mulf
// CHECK:   linalg.yield
// newAcc = P @ V + newAcc
// CHECK: linalg.generic
// CHECK:   arith.extf [[A:.+]] f8E4M3FNUZ to f32
// CHECK:   arith.extf [[A:.+]] f8E4M3FNUZ to f32
// CHECK:   arith.mulf
// CHECK:   arith.addf
// CHECK:   linalg.yield

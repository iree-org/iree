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

// -----

#score_payload_mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#score_payload_mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#score_payload_mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#score_payload_mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#score_payload_mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#score_payload_mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

func.func @online_attention_rank0_score_payload(
    %query: tensor<2x4x8xf32>,
    %key: tensor<2x5x8xf32>,
    %value: tensor<2x5x3xf32>,
    %output: tensor<2x4x3xf32>,
    %max: tensor<2x4xf32>,
    %sum: tensor<2x4xf32>)
    -> (tensor<2x4x3xf32>, tensor<2x4xf32>) {
  %scale = arith.constant 1.0 : f32

  %out:3 = iree_linalg_ext.online_attention
        { indexing_maps = [#score_payload_mapQ, #score_payload_mapK,
                           #score_payload_mapV, #score_payload_mapS,
                           #score_payload_mapO, #score_payload_mapR,
                           #score_payload_mapR] }
        ins(%query, %key, %value, %scale
            : tensor<2x4x8xf32>, tensor<2x5x8xf32>, tensor<2x5x3xf32>, f32)
        outs(%output, %max, %sum
            : tensor<2x4x3xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
      ^bb0(%score: f32):
        %score_tensor = tensor.from_elements %score : tensor<f32>
        %one = arith.constant dense<1.000000e+00> : tensor<f32>
        %empty = tensor.empty() : tensor<f32>
        %modified = linalg.generic {
            indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>,
                             affine_map<() -> ()>],
            iterator_types = []}
            ins(%score_tensor, %one : tensor<f32>, tensor<f32>)
            outs(%empty : tensor<f32>) {
          ^bb0(%in: f32, %bias: f32, %out: f32):
            %biased = arith.addf %in, %bias : f32
            %tanh = math.tanh %biased : f32
            linalg.yield %tanh : f32
        } -> tensor<f32>
        %extracted = tensor.extract %modified[] : tensor<f32>
        iree_linalg_ext.yield %extracted : f32
     }
        -> tensor<2x4x3xf32>, tensor<2x4xf32>, tensor<2x4xf32>

  return %out#0, %out#2 : tensor<2x4x3xf32>, tensor<2x4xf32>
}

// CHECK-LABEL: @online_attention_rank0_score_payload
// CHECK-NOT: tensor.from_elements
// CHECK-NOT: tensor.extract {{.*}}tensor<f32>
// CHECK: arith.addf
// CHECK: math.tanh
// CHECK-NOT: iterator_types = []
// CHECK: return

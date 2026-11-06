// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-aggregated-ops{filter-ops=iree_linalg_ext.online_attention}), canonicalize, cse)" --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-aggregated-ops{filter-ops=iree_linalg_ext.attention}), canonicalize, cse)" --split-input-file %s | FileCheck %s --check-prefix=ATTENTION

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
        < indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO, #mapR, #mapR] >
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

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

func.func @online_attention_dynamic_head(%query: tensor<4x6x?xf32>,
                         %key: tensor<4x10x?xf32>,
                         %value: tensor<4x10x8xf32>,
                         %output: tensor<4x6x8xf32>,
                         %max: tensor<4x6xf32>,
                         %sum: tensor<4x6xf32>)
                         -> (tensor<4x6x8xf32>, tensor<4x6xf32>) {
  %scale = arith.constant 0.125 : f32

  %out:3 = iree_linalg_ext.online_attention
        < indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO, #mapR, #mapR] >
        ins(%query, %key, %value, %scale : tensor<4x6x?xf32>, tensor<4x10x?xf32>, tensor<4x10x8xf32>, f32)
        outs(%output, %max, %sum : tensor<4x6x8xf32>, tensor<4x6xf32>, tensor<4x6xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     }
        -> tensor<4x6x8xf32>, tensor<4x6xf32>, tensor<4x6xf32>

  return %out#0, %out#2 : tensor<4x6x8xf32>, tensor<4x6xf32>
}
// A scaled copy of a dynamic query would need a buffer of the head's size, so
// the scale applies to S.
// CHECK-LABEL: @online_attention_dynamic_head
// CHECK-SAME:    %[[QUERY:[a-zA-Z0-9]+]]: tensor<4x6x?xf32>
// CHECK-SAME:    %[[KEY:[a-zA-Z0-9]+]]: tensor<4x10x?xf32>
// S = Q @ K
// CHECK:       %[[S:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[QUERY]], %[[KEY]] : tensor<4x6x?xf32>, tensor<4x10x?xf32>)
// CHECK:         arith.mulf
// CHECK:         arith.addf
// S = S * scale
// CHECK:       linalg.generic
// CHECK-SAME:    outs(%[[S]] : tensor<4x6x10xf32>)
// CHECK:         arith.mulf
// CHECK-NEXT:    linalg.yield
// newMax = max(oldMax, rowMax(S))
// CHECK:       linalg.generic
// CHECK:         arith.maximumf

// -----

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapM = affine_map<(batch, m, k1, k2, n) -> (batch, k2)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

// Every query row sees the same keys, so the excluded rows of V are zeroed
// once and P @ V stays a contraction.
func.func @online_attention_broadcast_mask(%query: tensor<4x6x16xf32>, %key: tensor<4x10x16xf32>,
    %value: tensor<4x10x8xf32>, %mask: tensor<4x10xi1>, %output: tensor<4x6x8xf32>,
    %max: tensor<4x6xf32>, %sum: tensor<4x6xf32>) -> tensor<4x6x8xf32> {
  %scale = arith.constant 0.125 : f32
  %out:3 = iree_linalg_ext.online_attention
        < indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO, #mapR, #mapR] >
        ins(%query, %key, %value, %scale, %mask : tensor<4x6x16xf32>, tensor<4x10x16xf32>, tensor<4x10x8xf32>, f32, tensor<4x10xi1>)
        outs(%output, %max, %sum : tensor<4x6x8xf32>, tensor<4x6xf32>, tensor<4x6xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     }
        -> tensor<4x6x8xf32>, tensor<4x6xf32>, tensor<4x6xf32>
  return %out#0 : tensor<4x6x8xf32>
}
// CHECK-LABEL: @online_attention_broadcast_mask
// CHECK-SAME:    %[[VALUE:[a-zA-Z0-9]+]]: tensor<4x10x8xf32>
// CHECK-SAME:    %[[MASK:[a-zA-Z0-9]+]]:
// CHECK:       %[[SAFE:.+]] = linalg.generic {{.*}} ins(%[[MASK]], %[[VALUE]] :
// CHECK:         arith.select %{{.+}}, %{{.+}}, %{{.+}} : f32
// CHECK:       linalg.generic {{.*}} ins(%{{.+}}, %[[SAFE]] :
// CHECK:         arith.mulf
// CHECK:         arith.addf

// -----

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapM = affine_map<(batch, m, k1, k2, n) -> (batch, m, k2)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

// A one-row tile cannot see different keys in different rows.
func.func @online_attention_single_row_mask(%query: tensor<4x1x16xf32>, %key: tensor<4x10x16xf32>,
    %value: tensor<4x10x8xf32>, %mask: tensor<4x1x10xi1>, %output: tensor<4x1x8xf32>,
    %max: tensor<4x1xf32>, %sum: tensor<4x1xf32>) -> tensor<4x1x8xf32> {
  %scale = arith.constant 0.125 : f32
  %out:3 = iree_linalg_ext.online_attention
        < indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO, #mapR, #mapR] >
        ins(%query, %key, %value, %scale, %mask : tensor<4x1x16xf32>, tensor<4x10x16xf32>, tensor<4x10x8xf32>, f32, tensor<4x1x10xi1>)
        outs(%output, %max, %sum : tensor<4x1x8xf32>, tensor<4x1xf32>, tensor<4x1xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     }
        -> tensor<4x1x8xf32>, tensor<4x1xf32>, tensor<4x1xf32>
  return %out#0 : tensor<4x1x8xf32>
}
// CHECK-LABEL: @online_attention_single_row_mask
// CHECK-SAME:    %[[VALUE:[a-zA-Z0-9]+]]: tensor<4x10x8xf32>
// CHECK-SAME:    %[[MASK:[a-zA-Z0-9]+]]:
// CHECK:       %[[SAFE:.+]] = linalg.generic {{.*}} ins(%[[MASK]], %[[VALUE]] :
// CHECK:         arith.select
// CHECK:       linalg.generic {{.*}} ins(%{{.+}}, %[[SAFE]] :
// CHECK:         arith.mulf
// CHECK:         arith.addf

// -----

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapM = affine_map<(batch, m, k1, k2, n) -> (batch, k2)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

// A float mask excludes only at -inf; a finite bias leaves V as it is.
func.func @online_attention_broadcast_float_mask(%query: tensor<4x6x16xf32>, %key: tensor<4x10x16xf32>,
    %value: tensor<4x10x8xf32>, %mask: tensor<4x10xf32>, %output: tensor<4x6x8xf32>,
    %max: tensor<4x6xf32>, %sum: tensor<4x6xf32>) -> tensor<4x6x8xf32> {
  %scale = arith.constant 0.125 : f32
  %out:3 = iree_linalg_ext.online_attention
        < indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO, #mapR, #mapR] >
        ins(%query, %key, %value, %scale, %mask : tensor<4x6x16xf32>, tensor<4x10x16xf32>, tensor<4x10x8xf32>, f32, tensor<4x10xf32>)
        outs(%output, %max, %sum : tensor<4x6x8xf32>, tensor<4x6xf32>, tensor<4x6xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     }
        -> tensor<4x6x8xf32>, tensor<4x6xf32>, tensor<4x6xf32>
  return %out#0 : tensor<4x6x8xf32>
}
// CHECK-LABEL: @online_attention_broadcast_float_mask
// CHECK-SAME:    %[[VALUE:[a-zA-Z0-9]+]]: tensor<4x10x8xf32>
// CHECK-SAME:    %[[MASK:[a-zA-Z0-9]+]]:
// CHECK:       %[[SAFE:.+]] = linalg.generic {{.*}} ins(%[[MASK]], %[[VALUE]] :
// CHECK:         arith.cmpf oeq
// CHECK:         arith.select
// CHECK:       linalg.generic {{.*}} ins(%{{.+}}, %[[SAFE]] :
// CHECK:         arith.mulf
// CHECK:         arith.addf

// -----

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapM = affine_map<(batch, m, k1, k2, n) -> (batch, m, k2)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

// A key visible in one row and excluded in another cannot be zeroed in V.
func.func @online_attention_row_dependent_mask(%query: tensor<4x6x16xf32>, %key: tensor<4x10x16xf32>,
    %value: tensor<4x10x8xf32>, %mask: tensor<4x6x10xi1>, %output: tensor<4x6x8xf32>,
    %max: tensor<4x6xf32>, %sum: tensor<4x6xf32>) -> tensor<4x6x8xf32> {
  %scale = arith.constant 0.125 : f32
  %out:3 = iree_linalg_ext.online_attention
        < indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO, #mapR, #mapR] >
        ins(%query, %key, %value, %scale, %mask : tensor<4x6x16xf32>, tensor<4x10x16xf32>, tensor<4x10x8xf32>, f32, tensor<4x6x10xi1>)
        outs(%output, %max, %sum : tensor<4x6x8xf32>, tensor<4x6xf32>, tensor<4x6xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     }
        -> tensor<4x6x8xf32>, tensor<4x6xf32>, tensor<4x6xf32>
  return %out#0 : tensor<4x6x8xf32>
}
// CHECK-LABEL: @online_attention_row_dependent_mask
// CHECK-SAME:    %[[VALUE:[a-zA-Z0-9]+]]: tensor<4x10x8xf32>
// CHECK-NOT:   ins(%{{.+}}, %[[VALUE]] : tensor<4x6x10xi1>
// CHECK:       linalg.generic {{.*}} ins(%{{.+}}, %[[VALUE]] : tensor<4x6x10xf32>, tensor<4x10x8xf32>)
// CHECK:         arith.mulf
// CHECK:         arith.addf

// -----

// Scaling must precede the nonlinear score region, just as for static Q.
// CHECK-LABEL: func.func @online_attention_dynamic_nonlinear
// CHECK: %[[DOT:.+]] = {{linalg.generic|linalg.matmul}}
// CHECK: %[[SCALED:.+]] = linalg.generic
// CHECK-SAME: outs(%[[DOT]] : tensor<?x3xf32>)
// CHECK: arith.mulf
// CHECK: linalg.yield
// CHECK: linalg.generic
// CHECK-SAME: outs(%[[SCALED]] : tensor<?x3xf32>)
// CHECK: ^bb0(%[[SCORE:[a-zA-Z0-9_]+]]: f32):
// CHECK: %[[SQUARED:.+]] = arith.mulf %[[SCORE]], %[[SCORE]] : f32
// CHECK-NEXT: linalg.yield %[[SQUARED]] : f32
#q = affine_map<(m, d, k, n) -> (m, d)>
#k = affine_map<(m, d, k, n) -> (k, d)>
#v = affine_map<(m, d, k, n) -> (k, n)>
#s = affine_map<(m, d, k, n) -> ()>
#o = affine_map<(m, d, k, n) -> (m, n)>
#r = affine_map<(m, d, k, n) -> (m)>

func.func @online_attention_dynamic_nonlinear(%query: tensor<?x4xf32>, %key: tensor<3x4xf32>,
    %value: tensor<3x2xf32>, %output: tensor<?x2xf32>, %max: tensor<?xf32>, %sum: tensor<?xf32>) -> tensor<?x2xf32> {
  %scale = arith.constant 2.0 : f32
  %result:3 = iree_linalg_ext.online_attention
    <indexing_maps = [#q, #k, #v, #s, #o, #r, #r], decomposition_config = {use_exp2 = false}>
    ins(%query, %key, %value, %scale : tensor<?x4xf32>, tensor<3x4xf32>, tensor<3x2xf32>, f32)
    outs(%output, %max, %sum : tensor<?x2xf32>, tensor<?xf32>, tensor<?xf32>) {
    ^bb0(%score: f32):
      %squared = arith.mulf %score, %score : f32
      iree_linalg_ext.yield %squared : f32
    } -> tensor<?x2xf32>, tensor<?xf32>, tensor<?xf32>
  return %result#0 : tensor<?x2xf32>
}

// -----

// Scaling must precede the nonlinear score region, just as for static Q.
// ATTENTION-LABEL: func.func @attention_dynamic_nonlinear
// ATTENTION: %[[DOT:.+]] = {{linalg.generic|linalg.matmul}}
// ATTENTION: %[[SCALED:.+]] = linalg.generic
// ATTENTION-SAME: outs(%[[DOT]] : tensor<?x3xf32>)
// ATTENTION: arith.mulf
// ATTENTION: linalg.yield
// ATTENTION: linalg.generic
// ATTENTION-SAME: outs(%[[SCALED]] : tensor<?x3xf32>)
// ATTENTION: ^bb0(%[[SCORE:[a-zA-Z0-9_]+]]: f32):
// ATTENTION: %[[SQUARED:.+]] = arith.mulf %[[SCORE]], %[[SCORE]] : f32
// ATTENTION-NEXT: linalg.yield %[[SQUARED]] : f32
#q = affine_map<(m, d, k, n) -> (m, d)>
#k = affine_map<(m, d, k, n) -> (k, d)>
#v = affine_map<(m, d, k, n) -> (k, n)>
#s = affine_map<(m, d, k, n) -> ()>
#o = affine_map<(m, d, k, n) -> (m, n)>
#r = affine_map<(m, d, k, n) -> (m)>

func.func @attention_dynamic_nonlinear(%query: tensor<?x4xf32>, %key: tensor<3x4xf32>,
    %value: tensor<3x2xf32>, %output: tensor<?x2xf32>) -> tensor<?x2xf32> {
  %scale = arith.constant 2.0 : f32
  %result = iree_linalg_ext.attention
    <indexing_maps = [#q, #k, #v, #s, #o]>
    ins(%query, %key, %value, %scale : tensor<?x4xf32>, tensor<3x4xf32>, tensor<3x2xf32>, f32)
    outs(%output : tensor<?x2xf32>) {
    ^bb0(%score: f32):
      %squared = arith.mulf %score, %score : f32
      iree_linalg_ext.yield %squared : f32
    } -> tensor<?x2xf32>
  return %result : tensor<?x2xf32>
}

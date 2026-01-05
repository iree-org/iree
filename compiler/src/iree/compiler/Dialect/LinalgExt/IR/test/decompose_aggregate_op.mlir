// RUN: iree-opt --iree-transform-dialect-interpreter --canonicalize --mlir-print-local-scope --split-input-file %s | FileCheck %s

// Spec to decompose custom op.
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.custom_op"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.iree.decompose_aggregate_op %0 : (!transform.any_op) -> ()
    transform.yield
  }
}

func.func @custom_op_decomposition(%lhs1 : tensor<1000000x?xf32>,
    %rhs1 : tensor<?x?xf32>, %rhs2 : tensor<?x?xf32>, %scalar : f32,
    %outs1 : tensor<1000000x?xf32>, %outs2 : tensor<1000000x?xf32>)
    -> (tensor<1000000x?xf32>, tensor<1000000x?xf32>) {
  %0:2 = iree_linalg_ext.custom_op {
        indexing_maps = [affine_map<(d0, d1)[s0, s1] -> (d0, s0)>,
                         affine_map<(d0, d1)[s0, s1] -> (s0, s1)>,
                         affine_map<(d0, d1)[s0, s1] -> (s1, d1)>,
                         affine_map<(d0, d1)[s0, s1] -> ()>,
                         affine_map<(d0, d1)[s0, s1] -> (d0, s1)>,
                         affine_map<(d0, d1)[s0, s1] -> (d0, d1)>],
        iterator_types = [#iree_linalg_ext.iterator_type<parallel>,
                          #iree_linalg_ext.iterator_type<parallel>]}
        ins(%lhs1, %rhs1, %rhs2, %scalar
            : tensor<1000000x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, f32)
        outs(%outs1, %outs2 : tensor<1000000x?xf32>, tensor<1000000x?xf32>) {
      ^bb0(%t0 : tensor<?x?xf32>, %t1 : tensor<?x?xf32>, %t2 : tensor<?x?xf32>,
           %s : f32, %t3 : tensor<?x?xf32>, %t4 : tensor<?x?xf32>) :
        %0 = linalg.matmul ins(%t0, %t1 : tensor<?x?xf32>, tensor<?x?xf32>)
            outs(%t3 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %1 = linalg.matmul ins(%0, %t2 : tensor<?x?xf32>, tensor<?x?xf32>)
            outs(%t4 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %2 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                             affine_map<(d0, d1) -> ()>,
                             affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%1, %s : tensor<?x?xf32>, f32) outs(%1 : tensor<?x?xf32>) {
          ^bb0(%b0 : f32, %b1 : f32, %b2 :f32):
            %3 = arith.addf %b0, %b2 : f32
            linalg.yield %3 : f32
        } -> tensor<?x?xf32>
        iree_linalg_ext.yield %0, %2 : tensor<?x?xf32>, tensor<?x?xf32>
    } -> tensor<1000000x?xf32>, tensor<1000000x?xf32>
  return %0#0, %0#1 : tensor<1000000x?xf32>, tensor<1000000x?xf32>
}

// CHECK-LABEL: func @custom_op_decomposition(
//  CHECK-SAME:     %[[LHS1:[a-zA-Z0-9]+]]: tensor<1000000x?xf32>
//  CHECK-SAME:     %[[RHS1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[RHS2:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[SCALAR:[a-zA-Z0-9]+]]: f32
//  CHECK-SAME:     %[[INIT1:[a-zA-Z0-9]+]]: tensor<1000000x?xf32>
//  CHECK-SAME:     %[[INIT2:[a-zA-Z0-9]+]]: tensor<1000000x?xf32>
//       CHECK:   %[[MATMUL1:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[LHS1]], %[[RHS1]] :
//  CHECK-SAME:       outs(%[[INIT1]] :
//       CHECK:   %[[MATMUL2:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[MATMUL1]], %[[RHS2]] :
//  CHECK-SAME:       outs(%[[INIT2]] :
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[MATMUL2]], %[[SCALAR]] :
//  CHECK-SAME:       outs(%[[MATMUL2]] :
//       CHECK:   return %[[MATMUL1]], %[[GENERIC]]

// -----

// Spec to decompose online attention op.
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.iree.decompose_aggregate_op %0 : (!transform.any_op) -> ()
    transform.yield
  }
}

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

func.func @attention_f16(%query: tensor<192x1024x64xf16>,
                         %key: tensor<192x1024x64xf16>,
                         %value: tensor<192x1024x64xf16>,
                         %output: tensor<192x1024x64xf32>)
                         -> (tensor<192x1024x64xf32>) {
  %scale = arith.constant 1.0 : f16

  %out = iree_linalg_ext.attention
        { indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO] }
        ins(%query, %key, %value, %scale : tensor<192x1024x64xf16>, tensor<192x1024x64xf16>, tensor<192x1024x64xf16>, f16)
        outs(%output : tensor<192x1024x64xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     }
        -> tensor<192x1024x64xf32>

  return %out : tensor<192x1024x64xf32>
}

// We just want to check if we are using the correct algorithm
// CHECK-LABEL: @attention_f16
// Q = Q * scale
// CHECK: linalg.generic
// CHECK:   arith.mulf
// S = Q @ K
// CHECK: linalg.generic
// CHECK:   arith.extf
// CHECK:   arith.extf
// CHECK:   arith.mulf
// CHECK:   arith.addf
// CHECK:   linalg.yield
// max = rowMax(S)
// CHECK: linalg.generic
// CHECK-NOT: arith.extf
// CHECK:   arith.maximumf
// CHECK:   linalg.yield
// P = exp2(S - max)
// CHECK: linalg.generic
// CHECK-NOT: arith.extf
// CHECK:   arith.subf
// CHECK:   math.exp2
// CHECK:   linalg.yield
// sum = rowSum(P)
// CHECK: linalg.generic
// CHECK-NOT: arith.extf
// CHECK:   arith.addf
// CHECK:   linalg.yield
// P = P /= sum
// CHECK: linalg.generic
// CHECK-NOT: arith.extf
// CHECK:   arith.divf
// CHECK:   linalg.yield
// truncf P : f32 to f16
// CHECK: linalg.generic
// CHECK-NOT: arith.extf
// CHECK:   arith.truncf
// CHECK:   linalg.yield
// newAcc = P @ V
// CHECK: linalg.generic
// CHECK:   arith.extf
// CHECK:   arith.extf
// CHECK:   arith.mulf
// CHECK:   arith.addf
// CHECK:   linalg.yield

// -----

// Spec to decompose online attention op.
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.iree.decompose_aggregate_op %0 : (!transform.any_op) -> ()
    transform.yield
  }
}

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

func.func @online_attention_f16(%query: tensor<192x1024x64xf16>,
                         %key: tensor<192x1024x64xf16>,
                         %value: tensor<192x1024x64xf16>,
                         %output: tensor<192x1024x64xf32>,
                         %max: tensor<192x1024xf32>,
                         %sum: tensor<192x1024xf32>)
                         -> (tensor<192x1024x64xf32>, tensor<192x1024xf32>) {
  %scale = arith.constant 1.0 : f16

  %out:3 = iree_linalg_ext.online_attention
        { indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO, #mapR, #mapR] }
        ins(%query, %key, %value, %scale : tensor<192x1024x64xf16>, tensor<192x1024x64xf16>, tensor<192x1024x64xf16>, f16)
        outs(%output, %max, %sum : tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     }
        -> tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>

  return %out#0, %out#2 : tensor<192x1024x64xf32>, tensor<192x1024xf32>
}

// We just want to check if we are using the correct algorithm and the
// correct number of extf/truncfs are emitted.
// CHECK-LABEL: @online_attention_f16
// Q = Q * scale
// CHECK: linalg.generic
// CHECK:   arith.mulf
// S = Q @ K
// CHECK: linalg.generic
// CHECK:   arith.extf
// CHECK:   arith.extf
// CHECK:   arith.mulf
// CHECK:   arith.addf
// CHECK:   linalg.yield
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
// newAcc = norm * oldAcc
// CHECK: linalg.generic
// CHECK-NOT: arith.extf
// CHECK:   arith.mulf
// CHECK:   linalg.yield
// newAcc = P @ V + newAcc
// CHECK: linalg.generic
// CHECK:   arith.extf
// CHECK:   arith.extf
// CHECK:   arith.mulf
// CHECK:   arith.addf
// CHECK:   linalg.yield

// -----

// Spec to decompose online attention op.
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.iree.decompose_aggregate_op %0 : (!transform.any_op) -> ()
    transform.yield
  }
}

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

// Spec to decompose online attention op.
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.iree.decompose_aggregate_op %0 : (!transform.any_op) -> ()
    transform.yield
  }
}

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapM = affine_map<(batch, m, k1, k2, n) -> (batch, m, k2)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

func.func @online_attention_f8_masked(%query: tensor<192x1024x64xf8E4M3FNUZ>,
                              %key: tensor<192x1024x64xf8E4M3FNUZ>,
                              %value: tensor<192x1024x64xf8E4M3FNUZ>,
                              %mask: tensor<192x1024x1024xf8E4M3FNUZ>,
                              %output: tensor<192x1024x64xf32>,
                              %max: tensor<192x1024xf32>,
                              %sum: tensor<192x1024xf32>)
                              -> (tensor<192x1024x64xf32>, tensor<192x1024xf32>) {
  %scale = arith.constant 1.0 : f16

  %out:3 = iree_linalg_ext.online_attention
        { indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO, #mapR, #mapR] }
        ins(%query, %key, %value, %scale, %mask : tensor<192x1024x64xf8E4M3FNUZ>, tensor<192x1024x64xf8E4M3FNUZ>, tensor<192x1024x64xf8E4M3FNUZ>, f16, tensor<192x1024x1024xf8E4M3FNUZ>)
        outs(%output, %max, %sum : tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score: f32
                     }
        -> tensor<192x1024x64xf32>, tensor<192x1024xf32>, tensor<192x1024xf32>

  return %out#0, %out#2 : tensor<192x1024x64xf32>, tensor<192x1024xf32>
}
// CHECK-LABEL: @online_attention_f8_masked
// S = Q @ K
// CHECK: linalg.generic
// CHECK:   arith.extf %[[A:.+]] : f8E4M3FNUZ to f32
// CHECK:   arith.extf %[[A:.+]] : f8E4M3FNUZ to f32
// CHECK:   arith.mulf
// CHECK:   arith.addf
// CHECK:   linalg.yield
// S = S * scale
// CHECK:   linalg.generic
// CHECK:   arith.mulf
// S = S + mask
// CHECK:   arith.addf
// newMax = max(oldMax, rowMax(S))
// CHECK: linalg.generic
// CHECK:   arith.maximumf
// CHECK:   linalg.yield
// P = exp2(S - newMax)
// CHECK: linalg.generic
// CHECK:   arith.subf
// CHECK:   math.exp2
// CHECK:   linalg.yield
// norm = exp2(oldMax - newMax)
// CHECK: linalg.generic
// CHECK:   arith.subf
// CHECK:   math.exp2
// CHECK:   linalg.yield
// normSum = norm * oldSum
// CHECK: linalg.generic
// CHECK:   arith.mulf
// CHECK:   linalg.yield
// newSum = normSum + rowMax(P)
// CHECK: linalg.generic
// CHECK:   arith.addf
// CHECK:   linalg.yield

// -----

// Spec to decompose exp reduction op.
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.exp_reduction"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.iree.decompose_aggregate_op %0 : (!transform.any_op) -> ()
    transform.yield
  }
}

func.func @exp_reduction(
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
// CHECK-LABEL: @exp_reduction
// CHECK-SAME:    %[[S:[0-9A-Za-z]*]]: tensor<20x4096x4096xf32>
// CHECK-SAME:    %[[V:[0-9A-Za-z]*]]: tensor<20x4096x64xf32>
// CHECK:       %[[M:.*]]       = linalg.generic
// CHECK-SAME:                      ins(%[[S]]
// CHECK-SAME:                      outs(
// CHECK:                             arith.maximumf
// CHECK:       %[[sub:.*]]     = linalg.generic
// CHECK-SAME:                      ins(%[[M]]
// CHECK-SAME:                      outs(%[[S]]
// CHECK:                             arith.subf
// CHECK:                             math.exp2
// CHECK:       %[[n:.*]]       = linalg.generic
// CHECK-SAME:                      ins(%[[M]]
// CHECK:                             arith.subf
// CHECK:                             math.exp2
// CHECK:      %[[max_norm:.*]] = linalg.generic
// CHECK-SAME:                      ins(%[[n]]
// CHECK:                             arith.mulf
// CHECK:      %[[acc_norm:.*]] = linalg.generic
// CHECK-SAME:                      ins(%[[n]]
// CHECK:                             arith.mulf
// CHECK:      %[[SUM:.*]]      = linalg.generic
// CHECK-SAME:                      ins(%[[sub]]
// CHECK-SAME:                      outs(%[[max_norm]]
// CHECK:                             arith.addf
// CHECK:      %[[PV:.*]]       = linalg.generic
// CHECK-SAME:                      ins(%[[sub]], %[[V]]
// CHECK-SAME:                      outs(%[[acc_norm]]
// CHECK:                             arith.mulf
// CHECK:                             arith.addf
// CHECK:      return %[[M]], %[[SUM]], %[[PV]]

// -----

// Spec to decompose arg_compare op (argmax patterns).
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.arg_compare"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.iree.decompose_aggregate_op %0 : (!transform.any_op) -> ()
    transform.yield
  }
}

func.func @arg_compare_argmax_f32(%input: tensor<4x128xf32>,
                                   %out_val: tensor<4xf32>,
                                   %out_idx: tensor<4xi32>)
    -> (tensor<4xf32>, tensor<4xi32>) {
  %result:2 = iree_linalg_ext.arg_compare
      dimension(1)
      ins(%input : tensor<4x128xf32>)
      outs(%out_val, %out_idx : tensor<4xf32>, tensor<4xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<4xf32>, tensor<4xi32>
  return %result#0, %result#1 : tensor<4xf32>, tensor<4xi32>
}

// CHECK-LABEL: @arg_compare_argmax_f32
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9]+]]: tensor<4x128xf32>
// CHECK-SAME:    %[[OUT_VAL:[a-zA-Z0-9]+]]: tensor<4xf32>
// CHECK-SAME:    %[[OUT_IDX:[a-zA-Z0-9]+]]: tensor<4xi32>
// Fill with -inf for argmax (required by isArgmaxOp)
// CHECK:       %[[NEG_INF:.*]] = arith.constant 0xFF800000 : f32
// CHECK:       %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:       %[[FILL_VAL:.*]] = linalg.fill ins(%[[NEG_INF]] : f32) outs(%[[OUT_VAL]] : tensor<4xf32>)
// CHECK:       %[[FILL_IDX:.*]] = linalg.fill ins(%[[ZERO]] : i32) outs(%[[OUT_IDX]] : tensor<4xi32>)
// CHECK:       %[[GENERIC:.*]]:2 = linalg.generic
// CHECK-SAME:    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>]
// CHECK-SAME:    iterator_types = ["parallel", "reduction"]
// CHECK-SAME:    ins(%[[INPUT]] : tensor<4x128xf32>)
// CHECK-SAME:    outs(%[[FILL_VAL]], %[[FILL_IDX]] : tensor<4xf32>, tensor<4xi32>)
// CHECK:         linalg.index 1
// CHECK:         arith.index_cast
// CHECK:         arith.maximumf
// CHECK:         arith.cmpf ogt
// CHECK:         arith.select
// CHECK:         linalg.yield
// CHECK:       return %[[GENERIC]]#0, %[[GENERIC]]#1

// -----

// Spec to decompose arg_compare op (argmin patterns).
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.arg_compare"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.iree.decompose_aggregate_op %0 : (!transform.any_op) -> ()
    transform.yield
  }
}

func.func @arg_compare_argmin_f32(%input: tensor<4x128xf32>,
                                   %out_val: tensor<4xf32>,
                                   %out_idx: tensor<4xi32>)
    -> (tensor<4xf32>, tensor<4xi32>) {
  %result:2 = iree_linalg_ext.arg_compare
      dimension(1)
      ins(%input : tensor<4x128xf32>)
      outs(%out_val, %out_idx : tensor<4xf32>, tensor<4xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf olt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<4xf32>, tensor<4xi32>
  return %result#0, %result#1 : tensor<4xf32>, tensor<4xi32>
}

// CHECK-LABEL: @arg_compare_argmin_f32
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9]+]]: tensor<4x128xf32>
// CHECK-SAME:    %[[OUT_VAL:[a-zA-Z0-9]+]]: tensor<4xf32>
// CHECK-SAME:    %[[OUT_IDX:[a-zA-Z0-9]+]]: tensor<4xi32>
// CHECK:       %[[POS_INF:.*]] = arith.constant 0x7F800000 : f32
// CHECK:       %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:       %[[FILL_VAL:.*]] = linalg.fill ins(%[[POS_INF]] : f32) outs(%[[OUT_VAL]] : tensor<4xf32>)
// CHECK:       %[[FILL_IDX:.*]] = linalg.fill ins(%[[ZERO]] : i32) outs(%[[OUT_IDX]] : tensor<4xi32>)
// CHECK:       %[[GENERIC:.*]]:2 = linalg.generic
// CHECK-SAME:    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>]
// CHECK-SAME:    iterator_types = ["parallel", "reduction"]
// CHECK-SAME:    ins(%[[INPUT]] : tensor<4x128xf32>)
// CHECK-SAME:    outs(%[[FILL_VAL]], %[[FILL_IDX]] : tensor<4xf32>, tensor<4xi32>)
// CHECK:         linalg.index 1
// CHECK:         arith.index_cast
// CHECK:         arith.minimumf
// CHECK:         arith.cmpf olt
// CHECK:         arith.select
// CHECK:         linalg.yield
// CHECK:       return %[[GENERIC]]#0, %[[GENERIC]]#1

// -----

// Integer argmax with sgt predicate.
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.arg_compare"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.iree.decompose_aggregate_op %0 : (!transform.any_op) -> ()
    transform.yield
  }
}

func.func @arg_compare_argmax_i32(%input: tensor<4x128xi32>,
                                   %out_val: tensor<4xi32>,
                                   %out_idx: tensor<4xi32>)
    -> (tensor<4xi32>, tensor<4xi32>) {
  %result:2 = iree_linalg_ext.arg_compare
      dimension(1)
      ins(%input : tensor<4x128xi32>)
      outs(%out_val, %out_idx : tensor<4xi32>, tensor<4xi32>) {
    ^bb0(%a: i32, %b: i32):
      %cmp = arith.cmpi sgt, %a, %b : i32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<4xi32>, tensor<4xi32>
  return %result#0, %result#1 : tensor<4xi32>, tensor<4xi32>
}

// CHECK-LABEL: @arg_compare_argmax_i32
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9]+]]: tensor<4x128xi32>
// CHECK-SAME:    %[[OUT_VAL:[a-zA-Z0-9]+]]: tensor<4xi32>
// CHECK-SAME:    %[[OUT_IDX:[a-zA-Z0-9]+]]: tensor<4xi32>
// CHECK:       %[[MIN_INT:.*]] = arith.constant -2147483648 : i32
// CHECK:       %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:       %[[FILL_VAL:.*]] = linalg.fill ins(%[[MIN_INT]] : i32) outs(%[[OUT_VAL]] : tensor<4xi32>)
// CHECK:       %[[FILL_IDX:.*]] = linalg.fill ins(%[[ZERO]] : i32) outs(%[[OUT_IDX]] : tensor<4xi32>)
// CHECK:       %[[GENERIC:.*]]:2 = linalg.generic
// CHECK-SAME:    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>]
// CHECK-SAME:    iterator_types = ["parallel", "reduction"]
// CHECK-SAME:    ins(%[[INPUT]] : tensor<4x128xi32>)
// CHECK-SAME:    outs(%[[FILL_VAL]], %[[FILL_IDX]] : tensor<4xi32>, tensor<4xi32>)
// CHECK:         linalg.index 1
// CHECK:         arith.index_cast
// CHECK:         arith.maxsi
// CHECK:         arith.cmpi sgt
// CHECK:         arith.select
// CHECK:         linalg.yield
// CHECK:       return %[[GENERIC]]#0, %[[GENERIC]]#1

// -----

// Unsigned integer argmax with ugt predicate.
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["iree_linalg_ext.arg_compare"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.iree.decompose_aggregate_op %0 : (!transform.any_op) -> ()
    transform.yield
  }
}

func.func @arg_compare_argmax_u32(%input: tensor<4x128xi32>,
                                   %out_val: tensor<4xi32>,
                                   %out_idx: tensor<4xi32>)
    -> (tensor<4xi32>, tensor<4xi32>) {
  %result:2 = iree_linalg_ext.arg_compare
      dimension(1)
      ins(%input : tensor<4x128xi32>)
      outs(%out_val, %out_idx : tensor<4xi32>, tensor<4xi32>) {
    ^bb0(%a: i32, %b: i32):
      %cmp = arith.cmpi ugt, %a, %b : i32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<4xi32>, tensor<4xi32>
  return %result#0, %result#1 : tensor<4xi32>, tensor<4xi32>
}

// CHECK-LABEL: @arg_compare_argmax_u32
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9]+]]: tensor<4x128xi32>
// CHECK-SAME:    %[[OUT_VAL:[a-zA-Z0-9]+]]: tensor<4xi32>
// CHECK-SAME:    %[[OUT_IDX:[a-zA-Z0-9]+]]: tensor<4xi32>
// CHECK:       %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK:       %[[FILL_VAL:.*]] = linalg.fill ins(%[[ZERO]] : i32) outs(%[[OUT_VAL]] : tensor<4xi32>)
// CHECK:       %[[FILL_IDX:.*]] = linalg.fill ins(%[[ZERO]] : i32) outs(%[[OUT_IDX]] : tensor<4xi32>)
// CHECK:       %[[GENERIC:.*]]:2 = linalg.generic
// CHECK-SAME:    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>]
// CHECK-SAME:    iterator_types = ["parallel", "reduction"]
// CHECK-SAME:    ins(%[[INPUT]] : tensor<4x128xi32>)
// CHECK-SAME:    outs(%[[FILL_VAL]], %[[FILL_IDX]] : tensor<4xi32>, tensor<4xi32>)
// CHECK:         linalg.index 1
// CHECK:         arith.index_cast
// CHECK:         arith.maxui
// CHECK:         arith.cmpi ugt
// CHECK:         arith.select
// CHECK:         linalg.yield
// CHECK:       return %[[GENERIC]]#0, %[[GENERIC]]#1

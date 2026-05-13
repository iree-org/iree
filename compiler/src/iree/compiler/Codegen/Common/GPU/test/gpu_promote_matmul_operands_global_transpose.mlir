// RUN: iree-opt %s --split-input-file \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-promote-matmul-operands),canonicalize)" \
// RUN:   | FileCheck %s

// Verify UseGlobalTransposeLoad promotion:
//   - Creates a linalg.generic copy with K-inner thread mapping
//     (input map reads B[K,N], output map writes alloc[N,K])
//   - Tags the copy with #iree_gpu.use_global_transpose_load lowering config
//   - The matmul RHS is updated to use the promoted (transposed) buffer

// -----

#lowering_config = #iree_gpu.lowering_config<{
  promote_operands = [1],
  promotion_types = [#iree_gpu.use_global_transpose_load]}>

// CHECK-LABEL: func.func @transpose_promote_rhs
// CHECK-SAME:    %[[LHS:[A-Za-z0-9]+]]: tensor<32x64xbf16>
// CHECK-SAME:    %[[RHS:[A-Za-z0-9]+]]: tensor<64x128xbf16>
func.func @transpose_promote_rhs(%lhs: tensor<32x64xbf16>,
                                  %rhs: tensor<64x128xbf16>) -> tensor<32x128xbf16> {
  %cst = arith.constant 0.0 : bf16
  %empty = tensor.empty() : tensor<32x128xbf16>
  %fill = linalg.fill ins(%cst : bf16) outs(%empty : tensor<32x128xbf16>) -> tensor<32x128xbf16>
  %mm = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"],
    lowering_config = #lowering_config}
    ins(%lhs, %rhs : tensor<32x64xbf16>, tensor<64x128xbf16>)
    outs(%fill : tensor<32x128xbf16>) {
  ^bb0(%in0: bf16, %in1: bf16, %out0: bf16):
    %mul = arith.mulf %in0, %in1 : bf16
    %add = arith.addf %out0, %mul : bf16
    linalg.yield %add : bf16
  } -> tensor<32x128xbf16>
  return %mm : tensor<32x128xbf16>
}

// The copy linalg.generic has:
//   input  map: (d0, d1) -> (d1, d0)   reads B[K, N] (K-outer, N-inner)
//   output map: (d0, d1) -> (d0, d1)   writes alloc[N, K] (N-outer, K-inner)
// This gives K-inner thread assignment so global_load_tr's 8x8 wave transpose
// aligns with the thread→lane mapping.
//
// The copy has the transposed output shape [N=128, K=64] and K-inner config.
// CHECK:       tensor.empty() : tensor<128x64xbf16>
// CHECK:       linalg.generic
// CHECK-SAME:    ins({{.*}} : tensor<64x128xbf16>)
// CHECK-SAME:    outs({{.*}} : tensor<128x64xbf16>)
// CHECK-SAME:    lowering_config = #iree_gpu.use_global_transpose_load
//
// The matmul's RHS input is the promoted [N=128, K=64] buffer.
// CHECK:       linalg.generic
// CHECK-SAME:    ins({{.*}}, {{.*}} : tensor<32x64xbf16>, tensor<128x64xbf16>)

// -----

// LHS promotion with UseGlobalTransposeLoad (transposedLhs case).
#lowering_config_lhs = #iree_gpu.lowering_config<{
  promote_operands = [0],
  promotion_types = [#iree_gpu.use_global_transpose_load]}>

// CHECK-LABEL: func.func @transpose_promote_lhs
// CHECK-SAME:    %[[LHS:[A-Za-z0-9]+]]: tensor<64x32xbf16>
func.func @transpose_promote_lhs(%lhs: tensor<64x32xbf16>,
                                  %rhs: tensor<64x128xbf16>) -> tensor<32x128xbf16> {
  %cst = arith.constant 0.0 : bf16
  %empty = tensor.empty() : tensor<32x128xbf16>
  %fill = linalg.fill ins(%cst : bf16) outs(%empty : tensor<32x128xbf16>) -> tensor<32x128xbf16>
  // transposedLhs: LHS is K-outer (K, M) instead of (M, K)
  %mm = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"],
    lowering_config = #lowering_config_lhs}
    ins(%lhs, %rhs : tensor<64x32xbf16>, tensor<64x128xbf16>)
    outs(%fill : tensor<32x128xbf16>) {
  ^bb0(%in0: bf16, %in1: bf16, %out0: bf16):
    %mul = arith.mulf %in0, %in1 : bf16
    %add = arith.addf %out0, %mul : bf16
    linalg.yield %add : bf16
  } -> tensor<32x128xbf16>
  return %mm : tensor<32x128xbf16>
}

// CHECK:       linalg.generic
// CHECK-SAME:    ins({{.*}} : tensor<64x32xbf16>)
// CHECK-SAME:    lowering_config = #iree_gpu.use_global_transpose_load

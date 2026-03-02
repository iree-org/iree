// RUN: iree-opt %s --pass-pipeline='builtin.module(func.func(iree-gpu-distribute-inner-tiled-to-lanes, canonicalize, cse))' --split-input-file | FileCheck %s

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @distribute_VSMFMA_F32_16x16x32_F16(%lhs: tensor<8x32xf16>, %rhs: tensor<32x16xf16>, %acc: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.virtual_mma_layout<VSMFMA_F32_16x16x32_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<8x32xf16>, tensor<32x16xf16> into tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @distribute_VSMFMA_F32_16x16x32_F16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<8x32xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<32x16xf16>
//       CHECK:   scf.forall (%[[LANEID:.+]]) in (64) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<8x16xf32>)
//   CHECK-DAG:     %{{.+}}:4 = affine.delinearize_index %[[LANEID]] into (4, 8, 2)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]]{{.*}} [1, 8]
//   CHECK-DAG:     %{{.+}}:3 = affine.delinearize_index %[[LANEID]] into (4, 16)
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]]{{.*}} [8, 1]
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]]{{.*}} [2, 1]
//       CHECK:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_SLICE]], %[[RHS_SLICE]]) outs(%[[ACC_SLICE]])
//  CHECK-SAME:       kind = #iree_gpu.virtual_mma_layout<VSMFMA_F32_16x16x32_F16>
//  CHECK-SAME:       : tensor<1x8xf16>, tensor<8x1xf16> into tensor<2x1xf32>
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]

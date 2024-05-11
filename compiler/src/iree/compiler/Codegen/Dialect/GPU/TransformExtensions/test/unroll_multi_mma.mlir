// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @unroll_multi_mma_order(%lhs: vector<2x2x4xf16>, %rhs: vector<2x2x4xf16>, %acc: vector<2x2x4xf32>) -> vector<2x2x4xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
  } : vector<2x2x4xf16>, vector<2x2x4xf16> into vector<2x2x4xf32>
  return %0 : vector<2x2x4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.unroll_multi_mma
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @unroll_multi_mma_order
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<2x2x4xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<2x2x4xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<2x2x4xf32>

//       CHECK:   %[[DEST:.+]] = arith.constant dense<0.000000e+00> : vector<2x2x4xf32>
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [0, 0]
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [0, 0]
//       CHECK:   %[[ACC0:.+]] = vector.extract_strided_slice %[[ACC]] {offsets = [0, 0]
//       CHECK:   %[[MMA0_K0:.+]] = iree_gpu.multi_mma %{{.*}}, %{{.*}}, %[[ACC0]]
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [0, 0]
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [0, 1]
//       CHECK:   %[[ACC1:.+]] = vector.extract_strided_slice %[[ACC]] {offsets = [0, 1]
//       CHECK:   %[[MMA1_K0:.+]] = iree_gpu.multi_mma %{{.*}}, %{{.*}}, %[[ACC1]]
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [1, 0]
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [0, 0]
//       CHECK:   %[[ACC2:.+]] = vector.extract_strided_slice %[[ACC]] {offsets = [1, 0]
//       CHECK:   %[[MMA2_K0:.+]] = iree_gpu.multi_mma %{{.*}}, %{{.*}}, %[[ACC2]]
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [1, 0]
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [0, 1]
//       CHECK:   %[[ACC3:.+]] = vector.extract_strided_slice %[[ACC]] {offsets = [1, 1]
//       CHECK:   %[[MMA3_K0:.+]] = iree_gpu.multi_mma %{{.*}}, %{{.*}}, %[[ACC3]]
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [0, 1]
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [1, 0]
//       CHECK:   %[[MMA0:.+]] = iree_gpu.multi_mma %{{.*}}, %{{.*}}, %[[MMA0_K0]]
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [0, 1]
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [1, 1]
//       CHECK:   %[[MMA1:.+]] = iree_gpu.multi_mma %{{.*}}, %{{.*}}, %[[MMA1_K0]]
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [1, 1]
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [1, 0]
//       CHECK:   %[[MMA2:.+]] = iree_gpu.multi_mma %{{.*}}, %{{.*}}, %[[MMA2_K0]]
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [1, 1]
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [1, 1]
//       CHECK:   %[[MMA3:.+]] = iree_gpu.multi_mma %{{.*}}, %{{.*}}, %[[MMA3_K0]]
//       CHECK:   %[[IN0:.+]] = vector.insert_strided_slice %[[MMA0]], %[[DEST]] {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<1x1x4xf32> into vector<2x2x4xf32>
//       CHECK:   %[[IN1:.+]] = vector.insert_strided_slice %[[MMA1]], %[[IN0]] {offsets = [0, 1, 0]
//       CHECK:   %[[IN2:.+]] = vector.insert_strided_slice %[[MMA2]], %[[IN1]] {offsets = [1, 0, 0]
//       CHECK:   %[[RES:.+]] = vector.insert_strided_slice %[[MMA3]], %[[IN2]] {offsets = [1, 1, 0]
//       CHECK:   return %[[RES]]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @unroll_multi_mma_count(%lhs: vector<2x3x4xf16>, %rhs: vector<3x5x4xf16>, %acc: vector<2x5x4xf32>) -> vector<2x5x4xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
  } : vector<2x3x4xf16>, vector<3x5x4xf16> into vector<2x5x4xf32>
  return %0 : vector<2x5x4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.unroll_multi_mma
    } : !transform.any_op
    transform.yield
  }
}

//    CHECK-LABEL: func @unroll_multi_mma_count
// CHECK-COUNT-30:   %[[MMA:.+]] = iree_gpu.multi_mma {{.*}} : vector<1x1x4xf16>, vector<1x1x4xf16> into vector<1x1x4xf32>
// CHECK-COUNT-10:   vector.insert_strided_slice {{.*}} : vector<1x1x4xf32> into vector<2x5x4xf32>

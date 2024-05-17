// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_mfma_16x16x16(%lhs: vector<4xf16>, %rhs: vector<4xf16>, %acc: vector<4xf32>) -> vector<4xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
  } : vector<4xf16>, vector<4xf16> into vector<4xf32>
  return %0 : vector<4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_multi_mma
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_mfma_16x16x16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<4xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<4xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<4xf32>
//       CHECK:   amdgpu.mfma %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32
//  CHECK-SAME:     blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_mfma_32x32x8(%lhs: vector<4xf16>, %rhs: vector<4xf16>, %acc: vector<16xf32>) -> vector<16xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>
  } : vector<4xf16>, vector<4xf16> into vector<16xf32>
  return %0 : vector<16xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_multi_mma
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_mfma_32x32x8
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<4xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<4xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<16xf32>
//       CHECK:   amdgpu.mfma %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32
//  CHECK-SAME:     blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_wmma_16x16x16(%lhs: vector<16xf16>, %rhs: vector<16xf16>, %acc: vector<8xf32>) -> vector<8xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMA_F16_16x16x16_F32>
  } : vector<16xf16>, vector<16xf16> into vector<8xf32>
  return %0 : vector<8xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_multi_mma
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_wmma_16x16x16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<16xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<16xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<8xf32>
//       CHECK:   amdgpu.wmma %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     : vector<16xf16>, vector<16xf16>, vector<8xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_mfma_shape_cast_16x16x16(%lhs: vector<1x4xf16>, %rhs: vector<4x1xf16>, %acc: vector<4x1xf32>) -> vector<4x1xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
  } : vector<1x4xf16>, vector<4x1xf16> into vector<4x1xf32>
  return %0 : vector<4x1xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_multi_mma
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_mfma_shape_cast_16x16x16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<1x4xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<4x1xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<4x1xf32>
//   CHECK-DAG:   %[[LHSCAST:.+]] = vector.shape_cast %[[LHS]] : vector<1x4xf16> to vector<4xf16>
//   CHECK-DAG:   %[[RHSCAST:.+]] = vector.shape_cast %[[RHS]] : vector<4x1xf16> to vector<4xf16>
//   CHECK-DAG:   %[[ACCCAST:.+]] = vector.shape_cast %[[ACC]] : vector<4x1xf32> to vector<4xf32>
//       CHECK:   %[[MMA:.+]] = amdgpu.mfma %[[LHSCAST]] * %[[RHSCAST]] + %[[ACCCAST]]
//  CHECK-SAME:     blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32
//  CHECK-SAME:     blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//       CHECK:   vector.shape_cast %[[MMA]] : vector<4xf32> to vector<4x1xf32>

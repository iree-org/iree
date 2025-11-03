// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_mfma_16x16x16(%lhs: vector<4xf16>, %rhs: vector<4xf16>, %acc: vector<4xf32>) -> vector<4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<4xf16>, vector<4xf16> into vector<4xf32>
  return %0 : vector<4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_mfma_16x16x16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<4xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<4xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<4xf32>
//       CHECK:   amdgpu.mfma 16x16x16 %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_mfma_32x32x8(%lhs: vector<4xf16>, %rhs: vector<4xf16>, %acc: vector<16xf32>) -> vector<16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<4xf16>, vector<4xf16> into vector<16xf32>
  return %0 : vector<16xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_mfma_32x32x8
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<4xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<4xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<16xf32>
//       CHECK:   amdgpu.mfma 32x32x8 %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_col_major_multi_mma_mfma_32x32x8(%lhs: vector<4xf16>, %rhs: vector<4xf16>, %acc: vector<16xf32>) -> vector<16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16, col_major = true>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<4xf16>, vector<4xf16> into vector<16xf32>
  return %0 : vector<16xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_col_major_multi_mma_mfma_32x32x8
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<4xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<4xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<16xf32>
//       CHECK:   amdgpu.mfma 32x32x8 %[[RHS]] * %[[LHS]] + %[[ACC]]
//  CHECK-SAME:     blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>

// -----

#contraction_accesses = [
  affine_map<() -> ()>,
  affine_map<() -> ()>,
  affine_map<() -> ()>
]

func.func @lower_col_major_inner_tiled_virtual_16x16x32(%lhs: vector<8xf16>, %rhs: vector<8xf16>, %acc: vector<4xf32>) -> vector<4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.virtual_mma_layout<VMFMA_F32_16x16x32_F16, col_major = true>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<8xf16>, vector<8xf16> into vector<4xf32>
  return %0 : vector<4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_col_major_inner_tiled_virtual_16x16x32
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<8xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<8xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<4xf32>
//  CHECK: %[[LHS0:.*]] = vector.extract_strided_slice %[[LHS]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
//  CHECK: %[[RHS0:.*]] = vector.extract_strided_slice %[[RHS]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
//  CHECK: %[[ACC0:.*]] = amdgpu.mfma 16x16x16 %[[RHS0]] * %[[LHS0]] + %[[ACC]]
//  CHECK: %[[LHS1:.*]] = vector.extract_strided_slice %[[LHS]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
//  CHECK: %[[RHS1:.*]] = vector.extract_strided_slice %[[RHS]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
//  CHECK: %[[ACC1:.*]] = amdgpu.mfma 16x16x16 %[[RHS1]] * %[[LHS1]] + %[[ACC0]]
//  CHECK: return %[[ACC1]] : vector<4xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_wmmar3_16x16x16(%lhs: vector<16xf16>, %rhs: vector<16xf16>, %acc: vector<8xf32>) -> vector<8xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMAR3_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<16xf16>, vector<16xf16> into vector<8xf32>
  return %0 : vector<8xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_wmmar3_16x16x16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<16xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<16xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<8xf32>
//       CHECK:   amdgpu.wmma 16x16x16 %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     : vector<16xf16>, vector<16xf16>, vector<8xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_wmmar4_16x16x16(%lhs: vector<8xf16>, %rhs: vector<8xf16>, %acc: vector<8xf32>) -> vector<8xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<8xf16>, vector<8xf16> into vector<8xf32>
  return %0 : vector<8xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_wmmar4_16x16x16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<8xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<8xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<8xf32>
//       CHECK:   amdgpu.wmma 16x16x16 %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     : vector<8xf16>, vector<8xf16>, vector<8xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_wmma_f32_16x16x4_f32(%lhs: vector<2xf32>, %rhs: vector<2xf32>, %acc: vector<8xf32>) -> vector<8xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMA_F32_16x16x4_F32>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<2xf32>, vector<2xf32> into vector<8xf32>
  return %0 : vector<8xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_wmma_f32_16x16x4_f32
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<2xf32>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<2xf32>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<8xf32>
//       CHECK:   amdgpu.wmma 16x16x4 %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     : vector<2xf32>, vector<2xf32>, vector<8xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_wmma_f32_16x16x32_f16(%lhs: vector<16xf16>, %rhs: vector<16xf16>, %acc: vector<8xf32>) -> vector<8xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMA_F32_16x16x32_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<16xf16>, vector<16xf16> into vector<8xf32>
  return %0 : vector<8xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_wmma_f32_16x16x32_f16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<16xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<16xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<8xf32>
//       CHECK:   amdgpu.wmma 16x16x32 %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     : vector<16xf16>, vector<16xf16>, vector<8xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_wmma_f32_16x16x64_f8E4M3FN(%lhs: vector<32xf8E4M3FN>, %rhs: vector<32xf8E4M3FN>, %acc: vector<8xf32>) -> vector<8xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMA_F32_16x16x64_F8E4M3FN>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<32xf8E4M3FN>, vector<32xf8E4M3FN> into vector<8xf32>
  return %0 : vector<8xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_wmma_f32_16x16x64_f8E4M3FN
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<32xf8E4M3FN>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<32xf8E4M3FN>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<8xf32>
//       CHECK:   amdgpu.wmma 16x16x64 %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     : vector<32xf8E4M3FN>, vector<32xf8E4M3FN>, vector<8xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_wmma_f32_16x16x128_f8E4M3FN(%lhs: vector<64xf8E4M3FN>, %rhs: vector<64xf8E4M3FN>, %acc: vector<8xf32>) -> vector<8xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMA_F32_16x16x128_F8E4M3FN>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<64xf8E4M3FN>, vector<64xf8E4M3FN> into vector<8xf32>
  return %0 : vector<8xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_wmma_f32_16x16x128_f8E4M3FN
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<64xf8E4M3FN>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<64xf8E4M3FN>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<8xf32>
//       CHECK:   amdgpu.wmma 16x16x128 %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     : vector<64xf8E4M3FN>, vector<64xf8E4M3FN>, vector<8xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_wmma_f16_16x16x128_f8E4M3FN(%lhs: vector<64xf8E4M3FN>, %rhs: vector<64xf8E4M3FN>, %acc: vector<8xf16>) -> vector<8xf16> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMA_F16_16x16x128_F8E4M3FN>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<64xf8E4M3FN>, vector<64xf8E4M3FN> into vector<8xf16>
  return %0 : vector<8xf16>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_multi_mma_wmma_f16_16x16x128_f8E4M3FN
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<64xf8E4M3FN>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<64xf8E4M3FN>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<8xf16>
//       CHECK:   amdgpu.wmma 16x16x128 %[[LHS]] * %[[RHS]] + %[[ACC]]
//  CHECK-SAME:     : vector<64xf8E4M3FN>, vector<64xf8E4M3FN>, vector<8xf16>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_mfma_shape_cast_16x16x16(%lhs: vector<1x4xf16>, %rhs: vector<4x1xf16>, %acc: vector<4x1xf32>) -> vector<4x1xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<1x4xf16>, vector<4x1xf16> into vector<4x1xf32>
  return %0 : vector<4x1xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
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
//       CHECK:   %[[MMA:.+]] = amdgpu.mfma 16x16x16 %[[LHSCAST]] * %[[RHSCAST]] + %[[ACCCAST]]
//  CHECK-SAME:     blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//       CHECK:   vector.shape_cast %[[MMA]] : vector<4xf32> to vector<4x1xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_inner_tiled_mfma_scale_f32_16x16x128_b32(
      %lhs: vector<32xf4E2M1FN>, %rhs: vector<32xf8E4M3FN>, %lhsScale: vector<1xf8E8M0FNU>, %rhsScale: vector<1xf8E8M0FNU>,
      %acc: vector<4xf32>) -> vector<4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs, %lhsScale, %rhsScale) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<32xf4E2M1FN>, vector<32xf8E4M3FN>, vector<1xf8E8M0FNU>, vector<1xf8E8M0FNU> into vector<4xf32>
  return %0 : vector<4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_inner_tiled_mfma_scale_f32_16x16x128_b32
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<32xf4E2M1FN>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<32xf8E4M3FN>
//  CHECK-SAME:   %[[LHS_SCALE:[A-Za-z0-9]+]]: vector<1xf8E8M0FNU>
//  CHECK-SAME:   %[[RHS_SCALE:[A-Za-z0-9]+]]: vector<1xf8E8M0FNU>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<4xf32>
//  CHECK: %[[CST:.+]] = arith.constant dense<5.877470e-39> : vector<4xf8E8M0FNU>
//  CHECK: %[[LHS_SCALE_SCALAR:.+]] = vector.extract %[[LHS_SCALE]][0]
//  CHECK: %[[LHS_SCALE_LONG:.+]] = vector.insert %[[LHS_SCALE_SCALAR]], %[[CST]] [0]
//  CHECK: %[[RHS_SCALE_SCALAR:.+]] = vector.extract %[[RHS_SCALE]][0]
//  CHECK: %[[RHS_SCALE_LONG:.+]] = vector.insert %[[RHS_SCALE_SCALAR]], %[[CST]] [0]
//  CHECK: amdgpu.scaled_mfma 16x16x128 (%[[LHS_SCALE_LONG]][0] * %[[LHS]]) * (%[[RHS_SCALE_LONG]][0] * %[[RHS]]) + %[[ACC]]
//  CHECK-SAME: vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf8E4M3FN>, vector<4xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_inner_tiled_mfma_scale_f32_32x32x64_b32(
      %lhs: vector<32xf4E2M1FN>, %rhs: vector<32xf8E4M3FN>, %lhsScale: vector<1xf8E8M0FNU>, %rhsScale: vector<1xf8E8M0FNU>,
      %acc: vector<16xf32>) -> vector<16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs, %lhsScale, %rhsScale) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_32x32x64_B32,
      lhs_elem_type = f4E2M1FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<32xf4E2M1FN>, vector<32xf8E4M3FN>, vector<1xf8E8M0FNU>, vector<1xf8E8M0FNU> into vector<16xf32>
  return %0 : vector<16xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_inner_tiled_mfma_scale_f32_32x32x64_b32
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<32xf4E2M1FN>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<32xf8E4M3FN>
//  CHECK-SAME:   %[[LHS_SCALE:[A-Za-z0-9]+]]: vector<1xf8E8M0FNU>
//  CHECK-SAME:   %[[RHS_SCALE:[A-Za-z0-9]+]]: vector<1xf8E8M0FNU>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<16xf32>
//  CHECK: %[[CST:.+]] = arith.constant dense<5.877470e-39> : vector<4xf8E8M0FNU>
//  CHECK: %[[LHS_SCALE_SCALAR:.+]] = vector.extract %[[LHS_SCALE]][0]
//  CHECK: %[[LHS_SCALE_LONG:.+]] = vector.insert %[[LHS_SCALE_SCALAR]], %[[CST]] [0]
//  CHECK: %[[RHS_SCALE_SCALAR:.+]] = vector.extract %[[RHS_SCALE]][0]
//  CHECK: %[[RHS_SCALE_LONG:.+]] = vector.insert %[[RHS_SCALE_SCALAR]], %[[CST]] [0]
//  CHECK: amdgpu.scaled_mfma 32x32x64 (%[[LHS_SCALE_LONG]][0] * %[[LHS]]) * (%[[RHS_SCALE_LONG]][0] * %[[RHS]]) + %[[ACC]]
//  CHECK-SAME: vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf8E4M3FN>, vector<16xf32>

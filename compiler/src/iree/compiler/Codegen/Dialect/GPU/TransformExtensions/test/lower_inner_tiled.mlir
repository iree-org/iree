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
func.func @lower_col_major_multi_mma_wmmar3_16x16x16(%lhs: vector<16xf16>, %rhs: vector<16xf16>, %acc: vector<8xf32>) -> vector<8xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMAR3_F32_16x16x16_F16, col_major = true>,
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

// CHECK-LABEL: func @lower_col_major_multi_mma_wmmar3_16x16x16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<16xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<16xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<8xf32>
//       CHECK:   amdgpu.wmma 16x16x16 %[[RHS]] * %[[LHS]] + %[[ACC]]
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
func.func @lower_col_major_multi_mma_wmmar4_16x16x16(%lhs: vector<8xf16>, %rhs: vector<8xf16>, %acc: vector<8xf32>) -> vector<8xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16, col_major = true>,
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

// CHECK-LABEL: func @lower_col_major_multi_mma_wmmar4_16x16x16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<8xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<8xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<8xf32>
//       CHECK:   amdgpu.wmma 16x16x16 %[[RHS]] * %[[LHS]] + %[[ACC]]
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
//       CHECK:   %[[ACCCAST:.+]] = util.hoistable_conversion "shape_cast_to_intrinsic" inverts("shape_cast_from_intrinsic")
//  CHECK-SAME:     (%[[ACC_B:.+]] = %[[ACC]]) : (vector<4x1xf32>) -> vector<4xf32>
//       CHECK:     vector.shape_cast %[[ACC_B]] : vector<4x1xf32> to vector<4xf32>
//       CHECK:   %[[MMA:.+]] = amdgpu.mfma 16x16x16 %[[LHSCAST]] * %[[RHSCAST]] + %[[ACCCAST]]
//  CHECK-SAME:     blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
//       CHECK:   util.hoistable_conversion "shape_cast_from_intrinsic" inverts("shape_cast_to_intrinsic")
//  CHECK-SAME:     (%[[MMA_B:.+]] = %[[MMA]]) : (vector<4xf32>) -> vector<4x1xf32>
//       CHECK:     vector.shape_cast %[[MMA_B]] : vector<4xf32> to vector<4x1xf32>

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

// -----

// Test that block intrinsics with a single-element accumulator per thread
// (e.g. MFMA_F64_4x4x4x4B_F64) extract the acc to scalar before the mfma
// and broadcast the scalar result back to the original vector type.

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_multi_mma_mfma_f64_4x4x4x4b(%lhs: vector<1xf64>, %rhs: vector<1xf64>, %acc: vector<1xf64>) -> vector<1xf64> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F64_4x4x4x4B_F64>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<1xf64>, vector<1xf64> into vector<1xf64>
  return %0 : vector<1xf64>
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

// CHECK-LABEL: func @lower_multi_mma_mfma_f64_4x4x4x4b
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<1xf64>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<1xf64>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<1xf64>
//   CHECK-DAG:   %[[LHS_S:.+]] = vector.extract %[[LHS]][0] : f64 from vector<1xf64>
//   CHECK-DAG:   %[[RHS_S:.+]] = vector.extract %[[RHS]][0] : f64 from vector<1xf64>
//   CHECK-DAG:   %[[ACC_S:.+]] = vector.extract %[[ACC]][0] : f64 from vector<1xf64>
//       CHECK:   %[[MMA:.+]] = amdgpu.mfma 4x4x4 %[[LHS_S]] * %[[RHS_S]] + %[[ACC_S]] {blocks = 4 : i32} blgp = none : f64, f64, f64
//       CHECK:   vector.broadcast %[[MMA]] : f64 to vector<1xf64>

// -----
// 16-bit VDMFMA variants (F16, BF16).

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_vdmfma_f16_8x16x64(%A: vector<8xf16>, %B: vector<16xf16>, %C: vector<2xf32>) -> vector<2xf32> {
  %0 = iree_codegen.inner_tiled ins(%A, %B) outs(%C) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x64_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<8xf16>, vector<16xf16> into vector<2xf32>
  return %0 : vector<2xf32>
}

func.func @lower_vdmfma_bf16_8x16x64(%A: vector<8xbf16>, %B: vector<16xbf16>, %C: vector<2xf32>) -> vector<2xf32> {
  %0 = iree_codegen.inner_tiled ins(%A, %B) outs(%C) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x64_BF16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<8xbf16>, vector<16xbf16> into vector<2xf32>
  return %0 : vector<2xf32>
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

// CHECK-LABEL: func @lower_vdmfma_f16_8x16x64
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: vector<8xf16>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: vector<16xf16>
//  CHECK-SAME:   %[[C:[A-Za-z0-9]+]]: vector<2xf32>
//  CHECK:    %[[ACC_EXPAND:.+]] = util.hoistable_conversion "vdmfma_interleave_acc" inverts("vdmfma_deinterleave_acc")
//  CHECK-SAME:   (%{{[^ ]+}} = %{{[^ )]+}}) : (vector<2xf32>) -> vector<4xf32>
//       CHECK:   %{{[^ ]+}} = vector.interleave
//       CHECK:   %[[LANE_ID:.+]] = gpu.lane_id
//       CHECK:   %[[LOW_BIT:.+]] = arith.andi %[[LANE_ID]]
//       CHECK:   %[[IS_ODD:.+]] = arith.cmpi ne, %[[LOW_BIT]]
//       CHECK:   %[[SPARSE_IDX:.+]] = arith.select %[[IS_ODD]]
//       CHECK:   %[[A_LO:.+]] = vector.extract_strided_slice %[[A]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
//       CHECK:   %[[B_INTLV_0:.+]] = vector.shuffle %[[B]], %[[B]] [0, 1, 8, 9, 2, 3, 10, 11] : vector<16xf16>, vector<16xf16>
//       CHECK:   %[[SMFMAC_0:.+]] = amdgpu.sparse_mfma 16x16x32 %[[A_LO]] * %[[B_INTLV_0]] + %[[ACC_EXPAND]] sparse(%[[SPARSE_IDX]]
//       CHECK:   %[[A_HI:.+]] = vector.extract_strided_slice %[[A]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
//       CHECK:   %[[B_INTLV_1:.+]] = vector.shuffle %[[B]], %[[B]] [4, 5, 12, 13, 6, 7, 14, 15] : vector<16xf16>, vector<16xf16>
//       CHECK:   %[[SMFMAC_1:.+]] = amdgpu.sparse_mfma 16x16x32 %[[A_HI]] * %[[B_INTLV_1]] + %[[SMFMAC_0]] sparse(%[[SPARSE_IDX]]
//       CHECK:   %[[ACC_COLLAPSE:.+]] = util.hoistable_conversion "vdmfma_deinterleave_acc" inverts("vdmfma_interleave_acc")
//  CHECK-SAME:   (%[[ACC_COLLAPSE_ARG:[^ ]+]] = %[[SMFMAC_1]]) : (vector<4xf32>) -> vector<2xf32>
//       CHECK:   %[[EVENS:.+]], %[[ODDS:.+]] = vector.deinterleave %[[ACC_COLLAPSE_ARG]] : vector<4xf32> -> vector<2xf32>
//       CHECK:   %[[RESULT:.+]] = arith.addf %[[EVENS]], %[[ODDS]]
//       CHECK:   util.return %[[RESULT]] : vector<2xf32>
//       CHECK:   return %[[ACC_COLLAPSE]] : vector<2xf32>

// CHECK-LABEL: func @lower_vdmfma_bf16_8x16x64
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: vector<8xbf16>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: vector<16xbf16>
//  CHECK-SAME:   %[[C:[A-Za-z0-9]+]]: vector<2xf32>
//       CHECK:   %[[ACC_EXPAND:.+]] = util.hoistable_conversion "vdmfma_interleave_acc" inverts("vdmfma_deinterleave_acc")
//  CHECK-SAME:   (%{{[^ ]+}} = %{{[^ )]+}}) : (vector<2xf32>) -> vector<4xf32>
//       CHECK:   %{{[^ ]+}} = vector.interleave
//       CHECK:   %[[LANE_ID:.+]] = gpu.lane_id
//       CHECK:   %[[LOW_BIT:.+]] = arith.andi %[[LANE_ID]]
//       CHECK:   %[[IS_ODD:.+]] = arith.cmpi ne, %[[LOW_BIT]]
//       CHECK:   %[[SPARSE_IDX:.+]] = arith.select %[[IS_ODD]]
//       CHECK:   %[[A_LO:.+]] = vector.extract_strided_slice %[[A]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xbf16> to vector<4xbf16>
//       CHECK:   %[[B_INTLV_0:.+]] = vector.shuffle %[[B]], %[[B]] [0, 1, 8, 9, 2, 3, 10, 11] : vector<16xbf16>, vector<16xbf16>
//       CHECK:   %[[SMFMAC_0:.+]] = amdgpu.sparse_mfma 16x16x32 %[[A_LO]] * %[[B_INTLV_0]] + %[[ACC_EXPAND]] sparse(%[[SPARSE_IDX]]
//       CHECK:   %[[A_HI:.+]] = vector.extract_strided_slice %[[A]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xbf16> to vector<4xbf16>
//       CHECK:   %[[B_INTLV_1:.+]] = vector.shuffle %[[B]], %[[B]] [4, 5, 12, 13, 6, 7, 14, 15] : vector<16xbf16>, vector<16xbf16>
//       CHECK:   %[[SMFMAC_1:.+]] = amdgpu.sparse_mfma 16x16x32 %[[A_HI]] * %[[B_INTLV_1]] + %[[SMFMAC_0]] sparse(%[[SPARSE_IDX]]
//       CHECK:   %[[ACC_COLLAPSE:.+]] = util.hoistable_conversion "vdmfma_deinterleave_acc" inverts("vdmfma_interleave_acc")
//  CHECK-SAME:   (%[[ACC_COLLAPSE_ARG:[^ ]+]] = %[[SMFMAC_1]]) : (vector<4xf32>) -> vector<2xf32>
//       CHECK:   %[[EVENS:.+]], %[[ODDS:.+]] = vector.deinterleave %[[ACC_COLLAPSE_ARG]] : vector<4xf32> -> vector<2xf32>
//       CHECK:   %[[RESULT:.+]] = arith.addf %[[EVENS]], %[[ODDS]]
//       CHECK:   util.return %[[RESULT]] : vector<2xf32>
//       CHECK:   return %[[ACC_COLLAPSE]] : vector<2xf32>

// -----
// 8-bit VDMFMA variants (I8, F8E5M2FNUZ, F8E4M3FNUZ).

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_vdmfma_i8_8x16x128(%A: vector<16xi8>, %B: vector<32xi8>, %C: vector<2xi32>) -> vector<2xi32> {
  %0 = iree_codegen.inner_tiled ins(%A, %B) outs(%C) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.virtual_mma_layout<VDMFMA_I32_8x16x128_I8>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<16xi8>, vector<32xi8> into vector<2xi32>
  return %0 : vector<2xi32>
}

func.func @lower_vdmfma_f8E5M2FNUZ_8x16x128(%A: vector<16xf8E5M2FNUZ>, %B: vector<32xf8E5M2FNUZ>, %C: vector<2xf32>) -> vector<2xf32> {
  %0 = iree_codegen.inner_tiled ins(%A, %B) outs(%C) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x128_F8E5M2FNUZ>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<16xf8E5M2FNUZ>, vector<32xf8E5M2FNUZ> into vector<2xf32>
  return %0 : vector<2xf32>
}

func.func @lower_vdmfma_f8E5M2FNUZ_f8E4M3FNUZ_8x16x128(%A: vector<16xf8E5M2FNUZ>, %B: vector<32xf8E4M3FNUZ>, %C: vector<2xf32>) -> vector<2xf32> {
  %0 = iree_codegen.inner_tiled ins(%A, %B) outs(%C) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x128_F8E5M2FNUZ_F8E4M3FNUZ>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<16xf8E5M2FNUZ>, vector<32xf8E4M3FNUZ> into vector<2xf32>
  return %0 : vector<2xf32>
}

func.func @lower_vdmfma_f8E4M3FNUZ_f8E5M2FNUZ_8x16x128(%A: vector<16xf8E4M3FNUZ>, %B: vector<32xf8E5M2FNUZ>, %C: vector<2xf32>) -> vector<2xf32> {
  %0 = iree_codegen.inner_tiled ins(%A, %B) outs(%C) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x128_F8E4M3FNUZ_F8E5M2FNUZ>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<16xf8E4M3FNUZ>, vector<32xf8E5M2FNUZ> into vector<2xf32>
  return %0 : vector<2xf32>
}

func.func @lower_vdmfma_f8E4M3FNUZ_8x16x128(%A: vector<16xf8E4M3FNUZ>, %B: vector<32xf8E4M3FNUZ>, %C: vector<2xf32>) -> vector<2xf32> {
  %0 = iree_codegen.inner_tiled ins(%A, %B) outs(%C) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x128_F8E4M3FNUZ>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<16xf8E4M3FNUZ>, vector<32xf8E4M3FNUZ> into vector<2xf32>
  return %0 : vector<2xf32>
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

// CHECK-LABEL: func @lower_vdmfma_i8_8x16x128
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: vector<16xi8>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: vector<32xi8>
//  CHECK-SAME:   %[[C:[A-Za-z0-9]+]]: vector<2xi32>
//       CHECK:   %[[ACC_EXPAND:.+]] = util.hoistable_conversion "vdmfma_interleave_acc" inverts("vdmfma_deinterleave_acc")
//  CHECK-SAME:   (%{{[^ ]+}} = %{{[^ )]+}}) : (vector<2xi32>) -> vector<4xi32>
//       CHECK:   %{{[^ ]+}} = vector.interleave
//       CHECK:   %[[LANE_ID:.+]] = gpu.lane_id
//       CHECK:   %[[LOW_BIT:.+]] = arith.andi %[[LANE_ID]]
//       CHECK:   %[[IS_ODD:.+]] = arith.cmpi ne, %[[LOW_BIT]]
//       CHECK:   %[[SPARSE_IDX:.+]] = arith.select %[[IS_ODD]]
//       CHECK:   %[[A_LO:.+]] = vector.extract_strided_slice %[[A]] {offsets = [0], sizes = [8], strides = [1]} : vector<16xi8> to vector<8xi8>
//       CHECK:   %[[B_INTLV_0:.+]] = vector.shuffle %[[B]], %[[B]] [0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23] : vector<32xi8>, vector<32xi8>
//       CHECK:   %[[SMFMAC_0:.+]] = amdgpu.sparse_mfma 16x16x64 %[[A_LO]] * %[[B_INTLV_0]] + %[[ACC_EXPAND]] sparse(%[[SPARSE_IDX]]
//       CHECK:   %[[A_HI:.+]] = vector.extract_strided_slice %[[A]] {offsets = [8], sizes = [8], strides = [1]} : vector<16xi8> to vector<8xi8>
//       CHECK:   %[[B_INTLV_1:.+]] = vector.shuffle %[[B]], %[[B]] [8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31] : vector<32xi8>, vector<32xi8>
//       CHECK:   %[[SMFMAC_1:.+]] = amdgpu.sparse_mfma 16x16x64 %[[A_HI]] * %[[B_INTLV_1]] + %[[SMFMAC_0]] sparse(%[[SPARSE_IDX]]
//       CHECK:   %[[ACC_COLLAPSE:.+]] = util.hoistable_conversion "vdmfma_deinterleave_acc" inverts("vdmfma_interleave_acc")
//  CHECK-SAME:   (%[[ACC_COLLAPSE_ARG:[^ ]+]] = %[[SMFMAC_1]]) : (vector<4xi32>) -> vector<2xi32>
//       CHECK:   %[[EVENS:.+]], %[[ODDS:.+]] = vector.deinterleave %[[ACC_COLLAPSE_ARG]] : vector<4xi32> -> vector<2xi32>
//       CHECK:   %[[RESULT:.+]] = arith.addi %[[EVENS]], %[[ODDS]]
//       CHECK:   util.return %[[RESULT]] : vector<2xi32>
//       CHECK:   return %[[ACC_COLLAPSE]] : vector<2xi32>

// CHECK-LABEL: func @lower_vdmfma_f8E5M2FNUZ_8x16x128
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: vector<16xf8E5M2FNUZ>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: vector<32xf8E5M2FNUZ>
//  CHECK-SAME:   %[[C:[A-Za-z0-9]+]]: vector<2xf32>
//       CHECK:   %[[ACC_EXPAND:.+]] = util.hoistable_conversion "vdmfma_interleave_acc" inverts("vdmfma_deinterleave_acc")
//  CHECK-SAME:   (%{{[^ ]+}} = %{{[^ )]+}}) : (vector<2xf32>) -> vector<4xf32>
//       CHECK:   %{{[^ ]+}} = vector.interleave
//       CHECK:   %[[LANE_ID:.+]] = gpu.lane_id
//       CHECK:   %[[LOW_BIT:.+]] = arith.andi %[[LANE_ID]]
//       CHECK:   %[[IS_ODD:.+]] = arith.cmpi ne, %[[LOW_BIT]]
//       CHECK:   %[[SPARSE_IDX:.+]] = arith.select %[[IS_ODD]]
//       CHECK:   %[[A_LO:.+]] = vector.extract_strided_slice %[[A]] {offsets = [0], sizes = [8], strides = [1]} : vector<16xf8E5M2FNUZ> to vector<8xf8E5M2FNUZ>
//       CHECK:   %[[B_INTLV_0:.+]] = vector.shuffle %[[B]], %[[B]] [0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23] : vector<32xf8E5M2FNUZ>, vector<32xf8E5M2FNUZ>
//       CHECK:   %[[SMFMAC_0:.+]] = amdgpu.sparse_mfma 16x16x64 %[[A_LO]] * %[[B_INTLV_0]] + %[[ACC_EXPAND]] sparse(%[[SPARSE_IDX]]
//       CHECK:   %[[A_HI:.+]] = vector.extract_strided_slice %[[A]] {offsets = [8], sizes = [8], strides = [1]} : vector<16xf8E5M2FNUZ> to vector<8xf8E5M2FNUZ>
//       CHECK:   %[[B_INTLV_1:.+]] = vector.shuffle %[[B]], %[[B]] [8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31] : vector<32xf8E5M2FNUZ>, vector<32xf8E5M2FNUZ>
//       CHECK:   %[[SMFMAC_1:.+]] = amdgpu.sparse_mfma 16x16x64 %[[A_HI]] * %[[B_INTLV_1]] + %[[SMFMAC_0]] sparse(%[[SPARSE_IDX]]
//       CHECK:   %[[ACC_COLLAPSE:.+]] = util.hoistable_conversion "vdmfma_deinterleave_acc" inverts("vdmfma_interleave_acc")
//  CHECK-SAME:   (%[[ACC_COLLAPSE_ARG:[^ ]+]] = %[[SMFMAC_1]]) : (vector<4xf32>) -> vector<2xf32>
//       CHECK:   %[[EVENS:.+]], %[[ODDS:.+]] = vector.deinterleave %[[ACC_COLLAPSE_ARG]] : vector<4xf32> -> vector<2xf32>
//       CHECK:   %[[RESULT:.+]] = arith.addf %[[EVENS]], %[[ODDS]]
//       CHECK:   util.return %[[RESULT]] : vector<2xf32>
//       CHECK:   return %[[ACC_COLLAPSE]] : vector<2xf32>

// CHECK-LABEL: func @lower_vdmfma_f8E5M2FNUZ_f8E4M3FNUZ_8x16x128
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: vector<16xf8E5M2FNUZ>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: vector<32xf8E4M3FNUZ>
//  CHECK-SAME:   %[[C:[A-Za-z0-9]+]]: vector<2xf32>
//       CHECK:   %[[ACC_EXPAND:.+]] = util.hoistable_conversion "vdmfma_interleave_acc" inverts("vdmfma_deinterleave_acc")
//  CHECK-SAME:   (%{{[^ ]+}} = %{{[^ )]+}}) : (vector<2xf32>) -> vector<4xf32>
//       CHECK:   %{{[^ ]+}} = vector.interleave
//       CHECK:   %[[LANE_ID:.+]] = gpu.lane_id
//       CHECK:   %[[LOW_BIT:.+]] = arith.andi %[[LANE_ID]]
//       CHECK:   %[[IS_ODD:.+]] = arith.cmpi ne, %[[LOW_BIT]]
//       CHECK:   %[[SPARSE_IDX:.+]] = arith.select %[[IS_ODD]]
//       CHECK:   %[[A_LO:.+]] = vector.extract_strided_slice %[[A]] {offsets = [0], sizes = [8], strides = [1]} : vector<16xf8E5M2FNUZ> to vector<8xf8E5M2FNUZ>
//       CHECK:   %[[B_INTLV_0:.+]] = vector.shuffle %[[B]], %[[B]] [0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23] : vector<32xf8E4M3FNUZ>, vector<32xf8E4M3FNUZ>
//       CHECK:   %[[SMFMAC_0:.+]] = amdgpu.sparse_mfma 16x16x64 %[[A_LO]] * %[[B_INTLV_0]] + %[[ACC_EXPAND]] sparse(%[[SPARSE_IDX]]
//       CHECK:   %[[A_HI:.+]] = vector.extract_strided_slice %[[A]] {offsets = [8], sizes = [8], strides = [1]} : vector<16xf8E5M2FNUZ> to vector<8xf8E5M2FNUZ>
//       CHECK:   %[[B_INTLV_1:.+]] = vector.shuffle %[[B]], %[[B]] [8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31] : vector<32xf8E4M3FNUZ>, vector<32xf8E4M3FNUZ>
//       CHECK:   %[[SMFMAC_1:.+]] = amdgpu.sparse_mfma 16x16x64 %[[A_HI]] * %[[B_INTLV_1]] + %[[SMFMAC_0]] sparse(%[[SPARSE_IDX]]
//       CHECK:   %[[ACC_COLLAPSE:.+]] = util.hoistable_conversion "vdmfma_deinterleave_acc" inverts("vdmfma_interleave_acc")
//  CHECK-SAME:   (%[[ACC_COLLAPSE_ARG:[^ ]+]] = %[[SMFMAC_1]]) : (vector<4xf32>) -> vector<2xf32>
//       CHECK:   %[[EVENS:.+]], %[[ODDS:.+]] = vector.deinterleave %[[ACC_COLLAPSE_ARG]] : vector<4xf32> -> vector<2xf32>
//       CHECK:   %[[RESULT:.+]] = arith.addf %[[EVENS]], %[[ODDS]]
//       CHECK:   util.return %[[RESULT]] : vector<2xf32>
//       CHECK:   return %[[ACC_COLLAPSE]] : vector<2xf32>

// CHECK-LABEL: func @lower_vdmfma_f8E4M3FNUZ_f8E5M2FNUZ_8x16x128
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: vector<16xf8E4M3FNUZ>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: vector<32xf8E5M2FNUZ>
//  CHECK-SAME:   %[[C:[A-Za-z0-9]+]]: vector<2xf32>
//       CHECK:   %[[ACC_EXPAND:.+]] = util.hoistable_conversion "vdmfma_interleave_acc" inverts("vdmfma_deinterleave_acc")
//  CHECK-SAME:   (%{{[^ ]+}} = %{{[^ )]+}}) : (vector<2xf32>) -> vector<4xf32>
//       CHECK:   %{{[^ ]+}} = vector.interleave
//       CHECK:   %[[LANE_ID:.+]] = gpu.lane_id
//       CHECK:   %[[LOW_BIT:.+]] = arith.andi %[[LANE_ID]]
//       CHECK:   %[[IS_ODD:.+]] = arith.cmpi ne, %[[LOW_BIT]]
//       CHECK:   %[[SPARSE_IDX:.+]] = arith.select %[[IS_ODD]]
//       CHECK:   %[[A_LO:.+]] = vector.extract_strided_slice %[[A]] {offsets = [0], sizes = [8], strides = [1]} : vector<16xf8E4M3FNUZ> to vector<8xf8E4M3FNUZ>
//       CHECK:   %[[B_INTLV_0:.+]] = vector.shuffle %[[B]], %[[B]] [0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23] : vector<32xf8E5M2FNUZ>, vector<32xf8E5M2FNUZ>
//       CHECK:   %[[SMFMAC_0:.+]] = amdgpu.sparse_mfma 16x16x64 %[[A_LO]] * %[[B_INTLV_0]] + %[[ACC_EXPAND]] sparse(%[[SPARSE_IDX]]
//       CHECK:   %[[A_HI:.+]] = vector.extract_strided_slice %[[A]] {offsets = [8], sizes = [8], strides = [1]} : vector<16xf8E4M3FNUZ> to vector<8xf8E4M3FNUZ>
//       CHECK:   %[[B_INTLV_1:.+]] = vector.shuffle %[[B]], %[[B]] [8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31] : vector<32xf8E5M2FNUZ>, vector<32xf8E5M2FNUZ>
//       CHECK:   %[[SMFMAC_1:.+]] = amdgpu.sparse_mfma 16x16x64 %[[A_HI]] * %[[B_INTLV_1]] + %[[SMFMAC_0]] sparse(%[[SPARSE_IDX]]
//       CHECK:   %[[ACC_COLLAPSE:.+]] = util.hoistable_conversion "vdmfma_deinterleave_acc" inverts("vdmfma_interleave_acc")
//  CHECK-SAME:   (%[[ACC_COLLAPSE_ARG:[^ ]+]] = %[[SMFMAC_1]]) : (vector<4xf32>) -> vector<2xf32>
//       CHECK:   %[[EVENS:.+]], %[[ODDS:.+]] = vector.deinterleave %[[ACC_COLLAPSE_ARG]] : vector<4xf32> -> vector<2xf32>
//       CHECK:   %[[RESULT:.+]] = arith.addf %[[EVENS]], %[[ODDS]]
//       CHECK:   util.return %[[RESULT]] : vector<2xf32>
//       CHECK:   return %[[ACC_COLLAPSE]] : vector<2xf32>

// CHECK-LABEL: func @lower_vdmfma_f8E4M3FNUZ_8x16x128
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: vector<16xf8E4M3FNUZ>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: vector<32xf8E4M3FNUZ>
//  CHECK-SAME:   %[[C:[A-Za-z0-9]+]]: vector<2xf32>
//       CHECK:   %[[ACC_EXPAND:.+]] = util.hoistable_conversion "vdmfma_interleave_acc" inverts("vdmfma_deinterleave_acc")
//  CHECK-SAME:   (%{{[^ ]+}} = %{{[^ )]+}}) : (vector<2xf32>) -> vector<4xf32>
//       CHECK:   %{{[^ ]+}} = vector.interleave
//       CHECK:   %[[LANE_ID:.+]] = gpu.lane_id
//       CHECK:   %[[LOW_BIT:.+]] = arith.andi %[[LANE_ID]]
//       CHECK:   %[[IS_ODD:.+]] = arith.cmpi ne, %[[LOW_BIT]]
//       CHECK:   %[[SPARSE_IDX:.+]] = arith.select %[[IS_ODD]]
//       CHECK:   %[[A_LO:.+]] = vector.extract_strided_slice %[[A]] {offsets = [0], sizes = [8], strides = [1]} : vector<16xf8E4M3FNUZ> to vector<8xf8E4M3FNUZ>
//       CHECK:   %[[B_INTLV_0:.+]] = vector.shuffle %[[B]], %[[B]] [0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23] : vector<32xf8E4M3FNUZ>, vector<32xf8E4M3FNUZ>
//       CHECK:   %[[SMFMAC_0:.+]] = amdgpu.sparse_mfma 16x16x64 %[[A_LO]] * %[[B_INTLV_0]] + %[[ACC_EXPAND]] sparse(%[[SPARSE_IDX]]
//       CHECK:   %[[A_HI:.+]] = vector.extract_strided_slice %[[A]] {offsets = [8], sizes = [8], strides = [1]} : vector<16xf8E4M3FNUZ> to vector<8xf8E4M3FNUZ>
//       CHECK:   %[[B_INTLV_1:.+]] = vector.shuffle %[[B]], %[[B]] [8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31] : vector<32xf8E4M3FNUZ>, vector<32xf8E4M3FNUZ>
//       CHECK:   %[[SMFMAC_1:.+]] = amdgpu.sparse_mfma 16x16x64 %[[A_HI]] * %[[B_INTLV_1]] + %[[SMFMAC_0]] sparse(%[[SPARSE_IDX]]
//       CHECK:   %[[ACC_COLLAPSE:.+]] = util.hoistable_conversion "vdmfma_deinterleave_acc" inverts("vdmfma_interleave_acc")
//  CHECK-SAME:   (%[[ACC_COLLAPSE_ARG:[^ ]+]] = %[[SMFMAC_1]]) : (vector<4xf32>) -> vector<2xf32>
//       CHECK:   %[[EVENS:.+]], %[[ODDS:.+]] = vector.deinterleave %[[ACC_COLLAPSE_ARG]] : vector<4xf32> -> vector<2xf32>
//       CHECK:   %[[RESULT:.+]] = arith.addf %[[EVENS]], %[[ODDS]]
//       CHECK:   util.return %[[RESULT]] : vector<2xf32>
//       CHECK:   return %[[ACC_COLLAPSE]] : vector<2xf32>

// RUN: iree-opt %s --pass-pipeline='builtin.module(func.func(iree-gpu-expand-undistributed-inner-tiles{expand-outputs=false}, canonicalize, cse))' --split-input-file | FileCheck %s -check-prefixes=CHECK,CHECK-INPUTS
// RUN: iree-opt %s --pass-pipeline='builtin.module(func.func(iree-gpu-expand-undistributed-inner-tiles{expand-inputs=false}, canonicalize, cse))' --split-input-file | FileCheck %s -check-prefixes=CHECK,CHECK-OUTPUTS

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 4], thread = [8, 4]}>
func.func @concretize_multi_mma_F32_16x16x16_F16(%lhs: tensor<2x2x16x16xf16>, %rhs: tensor<2x2x16x16xf16>, %acc: tensor<2x2x16x16xf32>) -> tensor<2x2x16x16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, lowering_config = #config
  } : tensor<2x2x16x16xf16>, tensor<2x2x16x16xf16> into tensor<2x2x16x16xf32>
  return %0 : tensor<2x2x16x16xf32>
}

// CHECK-LABEL:       func @concretize_multi_mma_F32_16x16x16_F16
// CHECK-SAME:          %[[LHS:[A-Za-z0-9]+]]: tensor<2x2x16x16xf16>
// CHECK-SAME:          %[[RHS:[A-Za-z0-9]+]]: tensor<2x2x16x16xf16>
// CHECK-SAME:          %[[ACC:[A-Za-z0-9]+]]: tensor<2x2x16x16xf32>

// CHECK-INPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-INPUTS-SAME:     lowering_config = #iree_gpu.lowering_config
// CHECK-INPUTS-SAME:     : tensor<2x2x16x16xf16>, tensor<2x2x16x16xf16> into tensor<2x2x16x16xf32>
// CHECK-INPUTS:        return %[[MMA]]

// CHECK-OUTPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-OUTPUTS-SAME:     lowering_config = #iree_gpu.lowering_config
// CHECK-OUTPUTS-SAME:     : tensor<2x2x16x16xf16>, tensor<2x2x16x16xf16> into tensor<2x2x16x16xf32>
// CHECK-OUTPUTS:        return %[[MMA]]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 4], thread = [8, 4]}>
func.func @concretize_multi_mma_I32_16x16x16_I8(%lhs: tensor<2x2x16x16xi8>, %rhs: tensor<2x2x16x16xi8>, %acc: tensor<2x2x16x16xi32>) -> tensor<2x2x16x16xi32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_I32_16x16x16_I8>, lowering_config = #config
  } : tensor<2x2x16x16xi8>, tensor<2x2x16x16xi8> into tensor<2x2x16x16xi32>
  return %0 : tensor<2x2x16x16xi32>
}

// CHECK-LABEL:       func @concretize_multi_mma_I32_16x16x16_I8
// CHECK-SAME:          %[[LHS:[A-Za-z0-9]+]]: tensor<2x2x16x16xi8>
// CHECK-SAME:          %[[RHS:[A-Za-z0-9]+]]: tensor<2x2x16x16xi8>
// CHECK-SAME:          %[[ACC:[A-Za-z0-9]+]]: tensor<2x2x16x16xi32>

// CHECK-INPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-INPUTS-SAME:     lowering_config = #iree_gpu.lowering_config
// CHECK-INPUTS-SAME:     : tensor<2x2x16x16xi8>, tensor<2x2x16x16xi8> into tensor<2x2x16x16xi32>
// CHECK-INPUTS:        return %[[MMA]]

// CHECK-OUTPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-OUTPUTS-SAME:     lowering_config = #iree_gpu.lowering_config
// CHECK-OUTPUTS-SAME:     : tensor<2x2x16x16xi8>, tensor<2x2x16x16xi8> into tensor<2x2x16x16xi32>
// CHECK-OUTPUTS:        return %[[MMA]]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]
#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 4], thread = [8, 4]}>
func.func @concretize_multi_mma_I32_16x16x32_I8(%lhs: tensor<2x2x16x32xi8>, %rhs: tensor<2x2x16x32xi8>, %acc: tensor<2x2x16x16xi32>) -> tensor<2x2x16x16xi32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
    permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>], lowering_config = #config
  } : tensor<2x2x16x32xi8>, tensor<2x2x16x32xi8> into tensor<2x2x16x16xi32>
  return %0 : tensor<2x2x16x16xi32>
}

// CHECK-LABEL:       func @concretize_multi_mma_I32_16x16x32_I8
// CHECK-SAME:          %[[LHS:[A-Za-z0-9]+]]: tensor<2x2x16x32xi8>
// CHECK-SAME:          %[[RHS:[A-Za-z0-9]+]]: tensor<2x2x16x32xi8>
// CHECK-SAME:          %[[ACC:[A-Za-z0-9]+]]: tensor<2x2x16x16xi32>

// CHECK-INPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-INPUTS-SAME:     lowering_config = #iree_gpu.lowering_config
// CHECK-INPUTS-SAME:     permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>]
// CHECK-INPUTS-SAME:     : tensor<2x2x16x32xi8>, tensor<2x2x16x32xi8> into tensor<2x2x16x16xi32>
// CHECK-INPUTS:        return %[[MMA]]

// CHECK-OUTPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-OUTPUTS-SAME:     lowering_config = #iree_gpu.lowering_config
// CHECK-OUTPUTS-SAME:     : tensor<2x2x16x32xi8>, tensor<2x2x16x32xi8> into tensor<2x2x16x16xi32>
// CHECK-OUTPUTS:        return %[[MMA]]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 4], thread = [8, 4]}>
func.func @concretize_multi_mma_F32_32x32x8_F16(%lhs: tensor<2x2x32x8xf16>, %rhs: tensor<2x2x8x32xf16>, %acc: tensor<2x2x32x32xf32>) -> tensor<2x2x32x32xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>, lowering_config = #config
  } : tensor<2x2x32x8xf16>, tensor<2x2x8x32xf16> into tensor<2x2x32x32xf32>
  return %0 : tensor<2x2x32x32xf32>
}

// CHECK-LABEL:       func @concretize_multi_mma_F32_32x32x8_F16
// CHECK-SAME:          %[[LHS:[A-Za-z0-9]+]]: tensor<2x2x32x8xf16>
// CHECK-SAME:          %[[RHS:[A-Za-z0-9]+]]: tensor<2x2x8x32xf16>
// CHECK-SAME:          %[[ACC:[A-Za-z0-9]+]]: tensor<2x2x32x32xf32>

// CHECK-INPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-INPUTS-SAME:     lowering_config = #iree_gpu.lowering_config
// CHECK-INPUTS-SAME:     : tensor<2x2x32x8xf16>, tensor<2x2x8x32xf16> into tensor<2x2x32x32xf32>
// CHECK-INPUTS:        return %[[MMA]]

// CHECK-OUTPUTS-DAG:    %[[EXPANDED_ACC:.+]] = tensor.expand_shape %[[ACC]] {{\[}}[0], [1], [2, 3], [4]] output_shape [2, 2, 4, 8, 32]
// CHECK-OUTPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[EXPANDED_ACC]])
// CHECK-OUTPUTS-SAME:     lowering_config = #iree_gpu.lowering_config
// CHECK-OUTPUTS-SAME:     : tensor<2x2x32x8xf16>, tensor<2x2x8x32xf16> into tensor<2x2x4x8x32xf32>
// CHECK-OUTPUTS:        %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMA]] {{\[}}[0], [1], [2, 3], [4]]
// CHECK-OUTPUTS:        return %[[COLLAPSED]]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 4], thread = [8, 4]}>
func.func @concretize_multi_mma_I32_32x32x8_I8(%lhs: tensor<2x2x32x8xi8>, %rhs: tensor<2x2x8x32xi8>, %acc: tensor<2x2x32x32xi32>) -> tensor<2x2x32x32xi32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_I32_32x32x8_I8>, lowering_config = #config
  } : tensor<2x2x32x8xi8>, tensor<2x2x8x32xi8> into tensor<2x2x32x32xi32>
  return %0 : tensor<2x2x32x32xi32>
}

// CHECK-LABEL:       func @concretize_multi_mma_I32_32x32x8_I8
// CHECK-SAME:          %[[LHS:[A-Za-z0-9]+]]: tensor<2x2x32x8xi8>
// CHECK-SAME:          %[[RHS:[A-Za-z0-9]+]]: tensor<2x2x8x32xi8>
// CHECK-SAME:          %[[ACC:[A-Za-z0-9]+]]: tensor<2x2x32x32xi32>

// CHECK-INPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-INPUTS-SAME:     lowering_config = #iree_gpu.lowering_config
// CHECK-INPUTS-SAME:     : tensor<2x2x32x8xi8>, tensor<2x2x8x32xi8> into tensor<2x2x32x32xi32>
// CHECK-INPUTS:        return %[[MMA]]

// CHECK-OUTPUTS-DAG:    %[[EXPANDED_ACC:.+]] = tensor.expand_shape %[[ACC]] {{\[}}[0], [1], [2, 3], [4]] output_shape [2, 2, 4, 8, 32]
// CHECK-OUTPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[EXPANDED_ACC]])
// CHECK-OUTPUTS-SAME:     lowering_config = #iree_gpu.lowering_config
// CHECK-OUTPUTS-SAME:     : tensor<2x2x32x8xi8>, tensor<2x2x8x32xi8> into tensor<2x2x4x8x32xi32>
// CHECK-OUTPUTS:        %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMA]] {{\[}}[0], [1], [2, 3], [4]]
// CHECK-OUTPUTS:        return %[[COLLAPSED]]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
#config = #iree_gpu.lowering_config<{workgroup = [64, 64, 0], reduction = [0, 0, 4], thread = [8, 4]}>
func.func @concretize_multi_mma_F32_32x32x8_F16(%lhs: tensor<2x2x32x8xf16>, %rhs: tensor<2x2x8x32xf16>, %acc: tensor<2x2x32x32xf32>) -> tensor<2x2x32x32xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>, lowering_config = #config,
    permutations = [array<i64: 0, 1>, array<i64: 0, 1>, array<i64: 1, 0>]
  } : tensor<2x2x32x8xf16>, tensor<2x2x8x32xf16> into tensor<2x2x32x32xf32>
  return %0 : tensor<2x2x32x32xf32>
}

// CHECK-LABEL:       func @concretize_multi_mma_F32_32x32x8_F16
// CHECK-SAME:          %[[LHS:[A-Za-z0-9]+]]: tensor<2x2x32x8xf16>
// CHECK-SAME:          %[[RHS:[A-Za-z0-9]+]]: tensor<2x2x8x32xf16>
// CHECK-SAME:          %[[ACC:[A-Za-z0-9]+]]: tensor<2x2x32x32xf32>

// CHECK-OUTPUTS-DAG:    %[[EXPANDED_ACC:.+]] = tensor.expand_shape %[[ACC]] {{\[}}[0], [1], [2], [3, 4]] output_shape [2, 2, 32, 4, 8]
// CHECK-OUTPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[EXPANDED_ACC]])
// CHECK-OUTPUTS-SAME:     lowering_config = #iree_gpu.lowering_config
// CHECK-OUTPUTS-SAME:     permutations = [array<i64: 0, 1>, array<i64: 0, 1>, array<i64: 2, 0, 1>]
// CHECK-OUTPUTS-SAME:     : tensor<2x2x32x8xf16>, tensor<2x2x8x32xf16> into tensor<2x2x32x4x8xf32>
// CHECK-OUTPUTS:        %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMA]] {{\[}}[0], [1], [2], [3, 4]]
// CHECK-OUTPUTS:        return %[[COLLAPSED]]

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @concretize_F32_16x16x4_F32(%lhs: tensor<16x4xf32>, %rhs: tensor<4x16xf32>, %acc: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>
  } : tensor<16x4xf32>, tensor<4x16xf32> into tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// CHECK-LABEL:       func @concretize_F32_16x16x4_F32

// CHECK-INPUTS-NOT:    tensor.expand_shape
// CHECK-INPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled
// CHECK-INPUTS:        return %[[MMA]]

// CHECK-OUTPUTS-NOT:    tensor.expand_shape
// CHECK-OUTPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled
// CHECK-OUTPUTS:        return %[[MMA]]

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @concretize_F32_16x16x32_F8E4M3FNUZ(%lhs: tensor<16x32xf8E4M3FNUZ>, %rhs: tensor<32x16xf8E4M3FNUZ>, %acc: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
  } : tensor<16x32xf8E4M3FNUZ>, tensor<32x16xf8E4M3FNUZ> into tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// CHECK-LABEL:       func @concretize_F32_16x16x32_F8E4M3FNUZ

// CHECK-INPUTS-NOT:    tensor.expand_shape
// CHECK-INPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled
// CHECK-INPUTS:        return %[[MMA]]

// CHECK-OUTPUTS-NOT:    tensor.expand_shape
// CHECK-OUTPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled
// CHECK-OUTPUTS:        return %[[MMA]]

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @concretize_I32_32x32x16_I8(%lhs: tensor<32x16xi8>, %rhs: tensor<16x32xi8>, %acc: tensor<32x32xi32>) -> tensor<32x32xi32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>
  } : tensor<32x16xi8>, tensor<16x32xi8> into tensor<32x32xi32>
  return %0 : tensor<32x32xi32>
}

// CHECK-LABEL:       func @concretize_I32_32x32x16_I8
// CHECK-SAME:          %[[LHS:[A-Za-z0-9]+]]: tensor<32x16xi8>
// CHECK-SAME:          %[[RHS:[A-Za-z0-9]+]]: tensor<16x32xi8>
// CHECK-SAME:          %[[ACC:[A-Za-z0-9]+]]: tensor<32x32xi32>

// CHECK-INPUTS-NOT:    tensor.expand_shape
// CHECK-INPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled
// CHECK-INPUTS:        return %[[MMA]]

// CHECK-OUTPUTS:        %[[EXPANDED_ACC:.+]] = tensor.expand_shape %[[ACC]] {{\[}}[0, 1], [2]] output_shape [4, 8, 32]
// CHECK-OUTPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[EXPANDED_ACC]])
// CHECK-OUTPUTS-SAME:     : tensor<32x16xi8>, tensor<16x32xi8> into tensor<4x8x32xi32>
// CHECK-OUTPUTS:        %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMA]] {{\[}}[0, 1], [2]]
// CHECK-OUTPUTS:        return %[[COLLAPSED]]

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @concretize_WMMAR3_F16_16x16x16_F16(%lhs: tensor<16x16xf16>, %rhs: tensor<16x16xf16>, %acc: tensor<16x16xf16>) -> tensor<16x16xf16> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMAR3_F16_16x16x16_F16>
  } : tensor<16x16xf16>, tensor<16x16xf16> into tensor<16x16xf16>
  return %0 : tensor<16x16xf16>
}

// CHECK-LABEL:       func @concretize_WMMAR3_F16_16x16x16_F16
// CHECK-SAME:          %[[LHS:[A-Za-z0-9]+]]: tensor<16x16xf16>
// CHECK-SAME:          %[[RHS:[A-Za-z0-9]+]]: tensor<16x16xf16>
// CHECK-SAME:          %[[ACC:[A-Za-z0-9]+]]: tensor<16x16xf16>

// CHECK-INPUTS-NOT:    tensor.expand_shape
// CHECK-INPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled
// CHECK-INPUTS:        return %[[MMA]]

// CHECK-OUTPUTS:        %[[EXPANDED_ACC:.+]] = tensor.expand_shape %[[ACC]] {{\[}}[0, 1], [2]] output_shape [16, 1, 16]
// CHECK-OUTPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[EXPANDED_ACC]])
// CHECK-OUTPUTS-SAME:     : tensor<16x16xf16>, tensor<16x16xf16> into tensor<16x1x16xf16>
// CHECK-OUTPUTS:        %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMA]] {{\[}}[0, 1], [2]]
// CHECK-OUTPUTS:        return %[[COLLAPSED]]

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @concretize_WMMAR3_I32_16x16x16_I8(%lhs: tensor<16x16xi8>, %rhs: tensor<16x16xi8>, %acc: tensor<16x16xi32>) -> tensor<16x16xi32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMAR3_I32_16x16x16_I8>
  } : tensor<16x16xi8>, tensor<16x16xi8> into tensor<16x16xi32>
  return %0 : tensor<16x16xi32>
}

// CHECK-LABEL:       func @concretize_WMMAR3_I32_16x16x16_I8
// CHECK-SAME:          %[[LHS:[A-Za-z0-9]+]]: tensor<16x16xi8>
// CHECK-SAME:          %[[RHS:[A-Za-z0-9]+]]: tensor<16x16xi8>
// CHECK-SAME:          %[[ACC:[A-Za-z0-9]+]]: tensor<16x16xi32>

// CHECK-INPUTS-NOT:    tensor.expand_shape
// CHECK-INPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled
// CHECK-INPUTS:        return %[[MMA]]

// CHECK-OUTPUTS:        %[[EXPANDED_ACC:.+]] = tensor.expand_shape %[[ACC]] {{\[}}[0, 1], [2]] output_shape [8, 2, 16]
// CHECK-OUTPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[EXPANDED_ACC]])
// CHECK-OUTPUTS-SAME:     : tensor<16x16xi8>, tensor<16x16xi8> into tensor<8x2x16xi32>
// CHECK-OUTPUTS:        %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMA]] {{\[}}[0, 1], [2]]
// CHECK-OUTPUTS:        return %[[COLLAPSED]]

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @concretize_WMMAR4_F16_16x16x16_F16(%lhs: tensor<16x16xf16>, %rhs: tensor<16x16xf16>, %acc: tensor<16x16xf16>) -> tensor<16x16xf16> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMAR4_F16_16x16x16_F16>
  } : tensor<16x16xf16>, tensor<16x16xf16> into tensor<16x16xf16>
  return %0 : tensor<16x16xf16>
}

// CHECK-LABEL:       func @concretize_WMMAR4_F16_16x16x16_F16
// CHECK-SAME:          %[[LHS:[A-Za-z0-9]+]]: tensor<16x16xf16>
// CHECK-SAME:          %[[RHS:[A-Za-z0-9]+]]: tensor<16x16xf16>
// CHECK-SAME:          %[[ACC:[A-Za-z0-9]+]]: tensor<16x16xf16>

// CHECK-INPUTS-NOT:    tensor.expand_shape
// CHECK-INPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled
// CHECK-INPUTS:        return %[[MMA]]

// CHECK-OUTPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-OUTPUTS-SAME:     : tensor<16x16xf16>, tensor<16x16xf16> into tensor<16x16xf16>
// CHECK-OUTPUTS:        return %[[MMA]]

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @concretize_WMMAR4_I32_16x16x16_I8(%lhs: tensor<16x16xi8>, %rhs: tensor<16x16xi8>, %acc: tensor<16x16xi32>) -> tensor<16x16xi32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMAR4_I32_16x16x16_I8>
  } : tensor<16x16xi8>, tensor<16x16xi8> into tensor<16x16xi32>
  return %0 : tensor<16x16xi32>
}

// CHECK-LABEL:       func @concretize_WMMAR4_I32_16x16x16_I8
// CHECK-SAME:          %[[LHS:[A-Za-z0-9]+]]: tensor<16x16xi8>
// CHECK-SAME:          %[[RHS:[A-Za-z0-9]+]]: tensor<16x16xi8>
// CHECK-SAME:          %[[ACC:[A-Za-z0-9]+]]: tensor<16x16xi32>

// CHECK-INPUTS-NOT:    tensor.expand_shape
// CHECK-INPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled
// CHECK-INPUTS:        return %[[MMA]]

// CHECK-OUTPUTS:        %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-OUTPUTS-SAME:     : tensor<16x16xi8>, tensor<16x16xi8> into tensor<16x16xi32>
// CHECK-OUTPUTS:        return %[[MMA]]

// -----

#contraction_accesses = [
 affine_map<(i, j, k, b) -> (i, k, b)>,
 affine_map<(i, j, k, b) -> (i, k)>,
 affine_map<(i, j, k, b) -> (k, b, j)>,
 affine_map<(i, j, k, b) -> (k, j)>,
 affine_map<(i, j, k, b) -> (i, j)>
]

func.func @expand_output_tile_scaled_mfma_32x32x64(%lhs: tensor<?x?x1x32x2x32xf4E2M1FN>, %lhsScale: tensor<?x?x32x2xf8E8M0FNU>,
    %rhs: tensor<?x1x?x32x2x32xf8E4M3FN>, %rhsScale: tensor<?x?x32x2xf8E8M0FNU>,
    %acc: tensor<?x?x32x32xf32>) -> tensor<?x?x32x32xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %lhsScale, %rhs, %rhsScale) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.scaled_mma_layout<
      intrinsic = MFMA_SCALE_F32_32x32x64_B32,
      lhs_elem_type = f4E2M1FN,
      rhs_elem_type = f8E4M3FN,
      acc_elem_type = f32>,
    permutations = [array<i64: 0, 1, 2>, array<i64: 0, 1>,
      array<i64: 2, 0, 1>, array<i64: 1, 0>,
      array<i64: 0, 1>]
  } : tensor<?x?x1x32x2x32xf4E2M1FN>, tensor<?x?x32x2xf8E8M0FNU>,
    tensor<?x1x?x32x2x32xf8E4M3FN>, tensor<?x?x32x2xf8E8M0FNU>
    into tensor<?x?x32x32xf32>
  return %0 : tensor<?x?x32x32xf32>
}

// CHECK-LABEL: func @expand_output_tile_scaled_mfma_32x32x64
// CHECK-SAME: %[[ACC:[A-Za-z0-9]+]]: tensor<?x?x32x32xf32>
// CHECK-INPUTS-NOT: tensor.expand_shape
// CHECK-INPUTS: return
// CHECK-OUTPUTS: tensor.expand_shape %[[ACC]]
// CHECK-OUTPUTS-SAME: tensor<?x?x4x8x32xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k, b) -> (i, k, b)>,
 affine_map<(i, j, k, b) -> (i, k)>,
 affine_map<(i, j, k, b) -> (k, b, j)>,
 affine_map<(i, j, k, b) -> (k, j)>,
 affine_map<(i, j, k, b) -> (i, j)>
]

func.func @expand_output_tile_scaled_mfma_32x32x64_col_major(%lhs: tensor<?x?x1x32x2x32xf4E2M1FN>, %lhsScale: tensor<?x?x32x2xf8E8M0FNU>,
    %rhs: tensor<?x1x?x32x2x32xf8E4M3FN>, %rhsScale: tensor<?x?x32x2xf8E8M0FNU>,
    %acc: tensor<?x?x32x32xf32>) -> tensor<?x?x32x32xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %lhsScale, %rhs, %rhsScale) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.scaled_mma_layout<
      intrinsic = MFMA_SCALE_F32_32x32x64_B32,
      lhs_elem_type = f4E2M1FN,
      rhs_elem_type = f8E4M3FN,
      acc_elem_type = f32, col_major = true>,
    permutations = [array<i64: 0, 1, 2>, array<i64: 0, 1>,
      array<i64: 2, 0, 1>, array<i64: 1, 0>,
      array<i64: 0, 1>]
  } : tensor<?x?x1x32x2x32xf4E2M1FN>, tensor<?x?x32x2xf8E8M0FNU>,
    tensor<?x1x?x32x2x32xf8E4M3FN>, tensor<?x?x32x2xf8E8M0FNU>
    into tensor<?x?x32x32xf32>
  return %0 : tensor<?x?x32x32xf32>
}

// CHECK-LABEL: func @expand_output_tile_scaled_mfma_32x32x64_col_major
// CHECK-SAME: %[[ACC:[A-Za-z0-9]+]]: tensor<?x?x32x32xf32>
// CHECK-INPUTS-NOT: tensor.expand_shape
// CHECK-OUTPUTS: tensor.expand_shape %[[ACC]]
// CHECK-OUTPUTS-SAME: tensor<?x?x32x4x8xf32>

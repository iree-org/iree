// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

// Basic case: expand_shape source of parallel_insert_slice triggers
// dest collapse, hoisting expand_shape out of the forall body.
func.func @collapse_dest_forall_basic(%buf: memref<4x2x8xf32>) {
  %empty = tensor.empty() : tensor<4x2x8xf32>
  %result = scf.forall (%i) = (0) to (4) step (1)
    shared_outs(%out = %empty) -> (tensor<4x2x8xf32>) {
    %slice = tensor.extract_slice %out[%i, 0, 0] [1, 2, 8] [1, 1, 1]
      : tensor<4x2x8xf32> to tensor<1x2x8xf32>
    %collapsed = tensor.collapse_shape %slice [[0], [1, 2]]
      : tensor<1x2x8xf32> into tensor<1x16xf32>
    %work = util.optimization_barrier %collapsed : tensor<1x16xf32>
    %expanded = tensor.expand_shape %work [[0], [1, 2]] output_shape [1, 2, 8]
      : tensor<1x16xf32> into tensor<1x2x8xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %expanded into %out[%i, 0, 0] [1, 2, 8] [1, 1, 1]
        : tensor<1x2x8xf32> into tensor<4x2x8xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %result, %buf
    : tensor<4x2x8xf32> into memref<4x2x8xf32>
  return
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.collapse_forall_dest
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @collapse_dest_forall_basic
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<4x2x8xf32>
//       CHECK:   %[[COLLAPSED_DEST:.+]] = tensor.collapse_shape %[[EMPTY]] {{\[}}[0], [1, 2]]
//  CHECK-SAME:     tensor<4x2x8xf32> into tensor<4x16xf32>
//       CHECK:   %[[FORALL:.+]] = scf.forall (%[[I:.+]]) in (4)
//  CHECK-SAME:     shared_outs(%[[OUT:.+]] = %[[COLLAPSED_DEST]]) -> (tensor<4x16xf32>)
//       CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[OUT]]
//  CHECK-SAME:       [%[[I]], {{.+}}] [1, 16] [1, 1]
//  CHECK-SAME:       tensor<4x16xf32> to tensor<1x16xf32>
//       CHECK:     %[[WORK:.+]] = util.optimization_barrier %[[EXTRACT]]
//       CHECK:     tensor.parallel_insert_slice %[[WORK]] into %[[OUT]]
//  CHECK-SAME:       [%[[I]], {{.+}}] [1, 16] [1, 1]
//  CHECK-SAME:       tensor<1x16xf32> into tensor<4x16xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[FORALL]] {{\[}}[0], [1, 2]]
//  CHECK-SAME:     tensor<4x16xf32> into tensor<4x2x8xf32>
//       CHECK:   iree_codegen.store_to_buffer %[[EXPANDED]], %{{.+}}

// -----

// Multiple results: only the tied result with expand_shape gets collapsed.
func.func @collapse_dest_forall_multiresult(
    %buf0: memref<32x32xf32>, %buf1: memref<4x8xf32>) {
  %empty0 = tensor.empty() : tensor<32x32xf32>
  %empty1 = tensor.empty() : tensor<4x8xf32>
  %result:2 = scf.forall (%i) = (0) to (4) step (1)
    shared_outs(%out0 = %empty0, %out1 = %empty1) -> (tensor<32x32xf32>, tensor<4x8xf32>) {
    %slice = tensor.extract_slice %out1[%i, 0] [1, 8] [1, 1]
      : tensor<4x8xf32> to tensor<1x8xf32>
    %collapsed = tensor.collapse_shape %slice [[0, 1]]
      : tensor<1x8xf32> into tensor<8xf32>
    %work = util.optimization_barrier %collapsed : tensor<8xf32>
    %expanded = tensor.expand_shape %work [[0, 1]] output_shape [1, 8]
      : tensor<8xf32> into tensor<1x8xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %out0 into %out0[0, 0] [32, 32] [1, 1]
        : tensor<32x32xf32> into tensor<32x32xf32>
      tensor.parallel_insert_slice %expanded into %out1[%i, 0] [1, 8] [1, 1]
        : tensor<1x8xf32> into tensor<4x8xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %result#0, %buf0
    : tensor<32x32xf32> into memref<32x32xf32>
  iree_codegen.store_to_buffer %result#1, %buf1
    : tensor<4x8xf32> into memref<4x8xf32>
  return
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.collapse_forall_dest
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @collapse_dest_forall_multiresult
//       CHECK:   %[[EMPTY0:.+]] = tensor.empty() : tensor<32x32xf32>
//       CHECK:   %[[EMPTY1:.+]] = tensor.empty() : tensor<4x8xf32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[EMPTY1]] {{\[}}[0, 1]]
//  CHECK-SAME:     tensor<4x8xf32> into tensor<32xf32>
//       CHECK:   %[[FORALL:.+]]:2 = scf.forall (%[[I:.+]]) in (4)
//  CHECK-SAME:     shared_outs(%[[OUT0:.+]] = %[[EMPTY0]], %[[OUT1:.+]] = %[[COLLAPSED]])
//  CHECK-SAME:     -> (tensor<32x32xf32>, tensor<32xf32>)
//       CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[OUT1]]
//  CHECK-SAME:       [{{.+}}] [8] [1]
//  CHECK-SAME:       tensor<32xf32> to tensor<8xf32>
//       CHECK:     %[[WORK:.+]] = util.optimization_barrier %[[EXTRACT]]
//       CHECK:     tensor.parallel_insert_slice %[[OUT0]] into %[[OUT0]]
//       CHECK:     tensor.parallel_insert_slice %[[WORK]] into %[[OUT1]]
//  CHECK-SAME:       [{{.+}}] [8] [1]
//  CHECK-SAME:       tensor<8xf32> into tensor<32xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[FORALL]]#1 {{\[}}[0, 1]]
//  CHECK-SAME:     tensor<32xf32> into tensor<4x8xf32>
//       CHECK:   iree_codegen.store_to_buffer %[[FORALL]]#0
//       CHECK:   iree_codegen.store_to_buffer %[[EXPANDED]]

// -----

// Rank-reducing parallel_insert_slice: expand_shape output rank < dest rank.
// The body produces data from an external input (no extract from dest),
// which avoids the reassociation mismatch between collapse and expand.
func.func @collapse_dest_forall_rank_reducing(
    %buf: memref<1x4x2x8xf32>, %input: tensor<4x16xf32>) {
  %empty = tensor.empty() : tensor<1x4x2x8xf32>
  %result = scf.forall (%i) = (0) to (4) step (1)
    shared_outs(%out = %empty) -> (tensor<1x4x2x8xf32>) {
    %slice_input = tensor.extract_slice %input[%i, 0] [1, 16] [1, 1]
      : tensor<4x16xf32> to tensor<1x16xf32>
    %work = util.optimization_barrier %slice_input : tensor<1x16xf32>
    %expanded = tensor.expand_shape %work [[0], [1, 2]] output_shape [1, 2, 8]
      : tensor<1x16xf32> into tensor<1x2x8xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %expanded into %out[0, %i, 0, 0] [1, 1, 2, 8] [1, 1, 1, 1]
        : tensor<1x2x8xf32> into tensor<1x4x2x8xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %result, %buf
    : tensor<1x4x2x8xf32> into memref<1x4x2x8xf32>
  return
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.collapse_forall_dest
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @collapse_dest_forall_rank_reducing
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<1x4x2x8xf32>
//       CHECK:   %[[COLLAPSED_DEST:.+]] = tensor.collapse_shape %[[EMPTY]] {{\[}}[0], [1], [2, 3]]
//  CHECK-SAME:     tensor<1x4x2x8xf32> into tensor<1x4x16xf32>
//       CHECK:   %[[FORALL:.+]] = scf.forall (%[[I:.+]]) in (4)
//  CHECK-SAME:     shared_outs(%[[OUT:.+]] = %[[COLLAPSED_DEST]]) -> (tensor<1x4x16xf32>)
//       CHECK:     %[[WORK:.+]] = util.optimization_barrier
//       CHECK:     tensor.parallel_insert_slice %[[WORK]] into %[[OUT]]
//  CHECK-SAME:       tensor<1x16xf32> into tensor<1x4x16xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[FORALL]] {{\[}}[0], [1], [2, 3]]
//  CHECK-SAME:     tensor<1x4x16xf32> into tensor<1x4x2x8xf32>
//       CHECK:   iree_codegen.store_to_buffer %[[EXPANDED]], %{{.+}}

// -----

// Non-trivial offset linearization: two induction variables index into
// dimensions that get collapsed, producing an affine.linearize_index.
func.func @collapse_dest_forall_linearize_offsets(%buf: memref<4x3x8xf32>) {
  %empty = tensor.empty() : tensor<4x3x8xf32>
  %result = scf.forall (%i, %j) = (0, 0) to (4, 3) step (1, 1)
    shared_outs(%out = %empty) -> (tensor<4x3x8xf32>) {
    %slice = tensor.extract_slice %out[%i, %j, 0] [1, 1, 8] [1, 1, 1]
      : tensor<4x3x8xf32> to tensor<1x1x8xf32>
    %collapsed = tensor.collapse_shape %slice [[0, 1], [2]]
      : tensor<1x1x8xf32> into tensor<1x8xf32>
    %work = util.optimization_barrier %collapsed : tensor<1x8xf32>
    %expanded = tensor.expand_shape %work [[0, 1], [2]] output_shape [1, 1, 8]
      : tensor<1x8xf32> into tensor<1x1x8xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %expanded into %out[%i, %j, 0] [1, 1, 8] [1, 1, 1]
        : tensor<1x1x8xf32> into tensor<4x3x8xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %result, %buf
    : tensor<4x3x8xf32> into memref<4x3x8xf32>
  return
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.collapse_forall_dest
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @collapse_dest_forall_linearize_offsets
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<4x3x8xf32>
//       CHECK:   %[[COLLAPSED_DEST:.+]] = tensor.collapse_shape %[[EMPTY]] {{\[}}[0, 1], [2]]
//  CHECK-SAME:     tensor<4x3x8xf32> into tensor<12x8xf32>
//       CHECK:   %[[FORALL:.+]] = scf.forall (%[[I:.+]], %[[J:.+]]) in (4, 3)
//  CHECK-SAME:     shared_outs(%[[OUT:.+]] = %[[COLLAPSED_DEST]]) -> (tensor<12x8xf32>)
//       CHECK:     %[[LIN:.+]] = affine.linearize_index disjoint [%[[I]], %[[J]]] by (4, 3)
//       CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[OUT]]
//  CHECK-SAME:       [%[[LIN]], {{.+}}] [1, 8] [1, 1]
//  CHECK-SAME:       tensor<12x8xf32> to tensor<1x8xf32>
//       CHECK:     %[[WORK:.+]] = util.optimization_barrier %[[EXTRACT]]
//       CHECK:     tensor.parallel_insert_slice %[[WORK]] into %[[OUT]]
//  CHECK-SAME:       [%[[LIN]], {{.+}}] [1, 8] [1, 1]
//  CHECK-SAME:       tensor<1x8xf32> into tensor<12x8xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[FORALL]] {{\[}}[0, 1], [2]]
//  CHECK-SAME:     tensor<12x8xf32> into tensor<4x3x8xf32>

// -----

// Negative test: dynamic insert sizes prevent collapsing.
// CHECK-LABEL: func @nocollapse_dest_forall_dynamic
func.func @nocollapse_dest_forall_dynamic(
    %buf: memref<4x?x8xf32>, %sz: index) {
  %empty = tensor.empty(%sz) : tensor<4x?x8xf32>
  %result = scf.forall (%i) = (0) to (4) step (1)
    shared_outs(%out = %empty) -> (tensor<4x?x8xf32>) {
    %slice = tensor.extract_slice %out[%i, 0, 0] [1, %sz, 8] [1, 1, 1]
      : tensor<4x?x8xf32> to tensor<1x?x8xf32>
    %collapsed = tensor.collapse_shape %slice [[0], [1, 2]]
      : tensor<1x?x8xf32> into tensor<1x?xf32>
    %work = util.optimization_barrier %collapsed : tensor<1x?xf32>
    %expanded = tensor.expand_shape %work [[0], [1, 2]] output_shape [1, %sz, 8]
      : tensor<1x?xf32> into tensor<1x?x8xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %expanded into %out[%i, 0, 0] [1, %sz, 8] [1, 1, 1]
        : tensor<1x?x8xf32> into tensor<4x?x8xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %result, %buf
    : tensor<4x?x8xf32> into memref<4x?x8xf32>
  return
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.collapse_forall_dest
    } : !transform.any_op
    transform.yield
  }
}

// Pattern should not fire: expand_shape remains inside the forall body.
//       CHECK:   scf.forall
//       CHECK:     tensor.expand_shape

// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

// Lifts a tensor-semantics `inner_tiled` to vector semantics: each operand
// becomes a `vector.transfer_read`, the op itself rebuilds with vector
// operands, and the result is written back through `vector.transfer_write`.
// The iteration domain (here trivially unit) is left untouched — the
// downstream `drop_inner_tiled_unit_dims` pattern will collapse it before the
// final intrinsic lowering fires.

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @vectorize_inner_tiled_avx512_f32(
    %lhs: tensor<1x1x1x1xf32>, %rhs: tensor<1x1x16x1xf32>,
    %acc: tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>,
                      #linalg.iterator_type<parallel>,
                      #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x16x1_F32_F32>,
    semantics = #iree_cpu.mma_semantics<>
  } : tensor<1x1x1x1xf32>, tensor<1x1x16x1xf32> into tensor<1x1x1x16xf32>
  return %0 : tensor<1x1x1x16xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.vectorize_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @vectorize_inner_tiled_avx512_f32
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9_]+]]: tensor<1x1x1x1xf32>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9_]+]]: tensor<1x1x16x1xf32>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9_]+]]: tensor<1x1x1x16xf32>
//   CHECK-DAG:   %[[VLHS:.+]] = vector.transfer_read %[[LHS]]{{.*}}: tensor<1x1x1x1xf32>, vector<1x1x1x1xf32>
//   CHECK-DAG:   %[[VRHS:.+]] = vector.transfer_read %[[RHS]]{{.*}}: tensor<1x1x16x1xf32>, vector<1x1x16x1xf32>
//   CHECK-DAG:   %[[VACC:.+]] = vector.transfer_read %[[ACC]]{{.*}}: tensor<1x1x1x16xf32>, vector<1x1x1x16xf32>
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled ins(%[[VLHS]], %[[VRHS]]) outs(%[[VACC]])
//  CHECK-SAME:       : vector<1x1x1x1xf32>, vector<1x1x16x1xf32> into vector<1x1x1x16xf32>
//       CHECK:   %[[WRITE:.+]] = vector.transfer_write %[[INNER]], %[[ACC]]
//       CHECK:   return %[[WRITE]]

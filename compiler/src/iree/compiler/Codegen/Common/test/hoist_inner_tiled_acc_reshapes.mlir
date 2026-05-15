// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-hoist-inner-tiled-acc-reshapes))" %s | FileCheck %s

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]

// CHECK-LABEL: @hoist_shape_cast_chain
// CHECK-SAME: %[[INIT:[a-zA-Z0-9]+]]: vector<2x2x1x1x4x1xf32>
// CHECK-DAG: %[[POISON:.+]] = ub.poison : vector<2x2x1x1x4x1xf32>
// CHECK-DAG: %[[SC0:.+]] = vector.shape_cast %[[INIT]]
// CHECK: %[[LOOP:.+]]:2 = scf.for {{.*}} iter_args(%[[DEADACC:.*]] = %[[POISON]], %[[ACC:.*]] = %[[SC0]])
// CHECK:   %[[OUT:.+]] = iree_codegen.inner_tiled {{.*}} outs(%[[ACC]])
// CHECK:   scf.yield %[[DEADACC]], %[[OUT]]
// CHECK: vector.shape_cast %[[LOOP]]#1
// CHECK-NOT: util.hoistable_conversion
func.func @hoist_shape_cast_chain(
    %lhs: vector<2x2x4xf16>, %rhs: vector<2x2x4xf16>,
    %init: vector<2x2x1x1x4x1xf32>) -> vector<2x2x1x1x4x1xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %result = scf.for %iv = %c0 to %c10 step %c1 iter_args(%acc = %init) -> vector<2x2x1x1x4x1xf32> {
    %inner_acc = vector.shape_cast %acc : vector<2x2x1x1x4x1xf32> to vector<2x2x4x1xf32>
    %mma = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%inner_acc) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>,
                        #linalg.iterator_type<parallel>,
                        #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
    } : vector<2x2x4xf16>, vector<2x2x4xf16> into vector<2x2x4x1xf32>
    %back = vector.shape_cast %mma : vector<2x2x4x1xf32> to vector<2x2x1x1x4x1xf32>
    scf.yield %back : vector<2x2x1x1x4x1xf32>
  }
  return %result : vector<2x2x1x1x4x1xf32>
}

// -----

#contraction_accesses2 = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]

// CHECK-LABEL: @no_reshape
// CHECK-NOT: util.hoistable_conversion
// CHECK-NOT: vector.shape_cast
func.func @no_reshape(
    %lhs: vector<2x2x4xf16>, %rhs: vector<2x2x4xf16>,
    %init: vector<2x2x4x1xf32>) -> vector<2x2x4x1xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %result = scf.for %iv = %c0 to %c10 step %c1 iter_args(%acc = %init) -> vector<2x2x4x1xf32> {
    %mma = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
      indexing_maps = #contraction_accesses2,
      iterator_types = [#linalg.iterator_type<parallel>,
                        #linalg.iterator_type<parallel>,
                        #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
      semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
    } : vector<2x2x4xf16>, vector<2x2x4xf16> into vector<2x2x4x1xf32>
    scf.yield %mma : vector<2x2x4x1xf32>
  }
  return %result : vector<2x2x4x1xf32>
}

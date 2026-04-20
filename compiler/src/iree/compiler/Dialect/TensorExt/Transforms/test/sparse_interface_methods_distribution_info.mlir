// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-tensor-ext-test-sparse-op-interface-methods{test-get-distribution-info=true}))' %s --split-input-file --allow-unregistered-dialect --mlir-print-local-scope --verify-diagnostics | FileCheck %s

// For ragged_dim(0): sparse dims are [0, 1].
// Dim 0 (outer) is distributable, dim 1 (inner) is not.
func.func public @distributionInfoRaggedDim0(
    %source : memref<?x?xf32>, %column_lengths: memref<?xi32>,
    %num_rows: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %source_d0 = memref.dim %source, %c0 : memref<?x?xf32>
  %source_d1 = memref.dim %source, %c1 : memref<?x?xf32>
  %0 = iree_tensor_ext.cast_to_ragged_shape %source
      ragged_dim(0) column_lengths(%column_lengths) num_ragged_rows(%num_rows)
      : (memref<?x?xf32>{%source_d0, %source_d1}, memref<?xi32>)
      -> memref<?x?x?xf32, #iree_tensor_ext.ragged_shape<0>>
  return
}
// CHECK-LABEL: func public @distributionInfoRaggedDim0
//       CHECK:   "test.distribution_info"() {distributable = array<i1: true, false>}

// -----

// For ragged_dim(1): sparse dims are [1, 2].
// Dim 1 (outer) is distributable, dim 2 (inner) is not.
func.func public @distributionInfoRaggedDim1(
    %source : memref<?x?x?xf32>, %column_lengths: memref<?xi32>,
    %num_rows: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %source_d0 = memref.dim %source, %c0 : memref<?x?x?xf32>
  %source_d1 = memref.dim %source, %c1 : memref<?x?x?xf32>
  %source_d2 = memref.dim %source, %c2 : memref<?x?x?xf32>
  %0 = iree_tensor_ext.cast_to_ragged_shape %source
      ragged_dim(1) column_lengths(%column_lengths) num_ragged_rows(%num_rows)
      : (memref<?x?x?xf32>{%source_d0, %source_d1, %source_d2}, memref<?xi32>)
      -> memref<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  return
}
// CHECK-LABEL: func public @distributionInfoRaggedDim1
//       CHECK:   "test.distribution_info"() {distributable = array<i1: true, false>}

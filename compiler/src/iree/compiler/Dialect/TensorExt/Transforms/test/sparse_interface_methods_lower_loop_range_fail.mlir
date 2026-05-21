// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-tensor-ext-test-sparse-op-interface-methods{test-lower-loop-range=true}, cse))' %s --split-input-file --allow-unregistered-dialect --mlir-print-local-scope --verify-diagnostics

// Test failure when only the inner sparse dimension is provided without the
// outer dimension. The inner loop depends on the outer IV for computing column
// bounds, so it cannot be lowered independently.
func.func public @innerWithoutOuter(%source : memref<?x?x?xf32>, %column_lengths: memref<?xi32>,
    %num_rows: index, %lb0 : index, %lb1 : index, %lb2 : index, %lb3 : index,
    %step0 : index, %step1 : index, %step2 : index, %step3 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %source_d0 = memref.dim %source, %c0 : memref<?x?x?xf32>
  %source_d1 = memref.dim %source, %c1 : memref<?x?x?xf32>
  %source_d2 = memref.dim %source, %c2 : memref<?x?x?xf32>
  // expected-error @+1 {{cannot lower inner sparse dimension without outer dimension}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source
      ragged_dim(1) column_lengths(%column_lengths) num_ragged_rows(%num_rows)
      : (memref<?x?x?xf32>{%source_d0, %source_d1, %source_d2}, memref<?xi32>)
      -> memref<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  %d0 = memref.dim %0, %c0 : memref<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  %d1 = memref.dim %0, %c1 : memref<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  %d2 = memref.dim %0, %c2 : memref<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  %d3 = memref.dim %0, %c3 : memref<?x?x?x?xf32, #iree_tensor_ext.ragged_shape<1>>
  scf.forall (%i, %j, %k, %l) = (%lb0, %lb1, %lb2, %lb3) to (%d0, %d1, %d2, %d3) step (%step0, %step1, %step2, %step3) {
    "some_op"(%i, %j, %k, %l) : (index, index, index, index) -> ()
  } {iree_tensor_ext.sparse_iteration_dims = #iree_tensor_ext.sparse_iteration_dims<[2]>}
  return
}

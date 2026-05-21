// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-tensor-ext-test-sparse-op-interface-methods{test-resolve-range=true}))' %s --split-input-file --allow-unregistered-dialect --mlir-print-local-scope --verify-diagnostics | FileCheck %s

// Test resolveRange for a simple 2D ragged tensor (memref).
// The test.range operands specify (offset, size, stride) triples for each
// result dimension:
//   dim 0 (outer sparse): offset=%row_idx, size=1, stride=1
//   dim 1 (inner sparse): offset=%col_offset, size=%col_size, stride=1
//   dim 2 (dense): offset=%dense_offset, size=%dense_size, stride=1
//
// Expected resolved range (2D source):
//   dim 0: offset = column_lengths[row_idx] + col_offset,
//          size = col_size, stride = 1
//   dim 1: offset = dense_offset, size = dense_size, stride = 1
func.func public @resolveRangeMemref(
    %source : memref<?x?xf32>, %column_lengths: memref<?xi32>,
    %num_rows: index, %row_idx: index, %col_offset: index,
    %col_size: index, %dense_offset: index, %dense_size: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %source_d0 = memref.dim %source, %c0 : memref<?x?xf32>
  %source_d1 = memref.dim %source, %c1 : memref<?x?xf32>
  %0 = iree_tensor_ext.cast_to_ragged_shape %source
      ragged_dim(0) column_lengths(%column_lengths) num_ragged_rows(%num_rows)
      : (memref<?x?xf32>{%source_d0, %source_d1}, memref<?xi32>)
      -> memref<?x?x?xf32, #iree_tensor_ext.ragged_shape<0>>
  "test.range"(%row_idx, %c1, %c1,
               %col_offset, %col_size, %c1,
               %dense_offset, %dense_size, %c1)
      : (index, index, index, index, index, index, index, index, index) -> ()
  return
}
// CHECK-LABEL: func public @resolveRangeMemref(
//  CHECK-SAME:     %[[SOURCE:.+]]: memref<?x?xf32>,
//  CHECK-SAME:     %[[COLUMN_LENGTHS:.+]]: memref<?xi32>,
//  CHECK-SAME:     %[[NUM_ROWS:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[ROW_IDX:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[COL_OFFSET:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[COL_SIZE:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[DENSE_OFFSET:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[DENSE_SIZE:[a-zA-Z0-9]+]]: index) {
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK:   %[[COL_LEN_I32:.+]] = memref.load %[[COLUMN_LENGTHS]][%[[ROW_IDX]]]
//       CHECK:   %[[COL_LEN:.+]] = arith.index_cast %[[COL_LEN_I32]] : i32 to index
//       CHECK:   %[[LINEARIZED:.+]] = affine.apply
//  CHECK-SAME:       affine_map<()[s0, s1] -> (s0 + s1)>()[%[[COL_LEN]], %[[COL_OFFSET]]]
//       CHECK:   "test.resolved_range"(%[[LINEARIZED]], %[[COL_SIZE]], %[[C1]],
//  CHECK-SAME:       %[[DENSE_OFFSET]], %[[DENSE_SIZE]], %[[C1]])

// -----

// Test resolveRange for a 2D ragged tensor with tensor types.
func.func public @resolveRangeTensor(
    %source : tensor<?x?xf32>, %column_lengths: tensor<?xi32>,
    %num_rows: index, %row_idx: index, %col_offset: index,
    %col_size: index, %dense_offset: index, %dense_size: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %source_d0 = tensor.dim %source, %c0 : tensor<?x?xf32>
  %source_d1 = tensor.dim %source, %c1 : tensor<?x?xf32>
  %0 = iree_tensor_ext.cast_to_ragged_shape %source
      ragged_dim(0) column_lengths(%column_lengths) num_ragged_rows(%num_rows)
      : (tensor<?x?xf32>{%source_d0, %source_d1}, tensor<?xi32>)
      -> tensor<?x?x?xf32, #iree_tensor_ext.ragged_shape<0>>
  "test.range"(%row_idx, %c1, %c1,
               %col_offset, %col_size, %c1,
               %dense_offset, %dense_size, %c1)
      : (index, index, index, index, index, index, index, index, index) -> ()
  return
}
// CHECK-LABEL: func public @resolveRangeTensor(
//  CHECK-SAME:     %[[SOURCE:.+]]: tensor<?x?xf32>,
//  CHECK-SAME:     %[[COLUMN_LENGTHS:.+]]: tensor<?xi32>,
//  CHECK-SAME:     %[[NUM_ROWS:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[ROW_IDX:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[COL_OFFSET:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[COL_SIZE:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[DENSE_OFFSET:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[DENSE_SIZE:[a-zA-Z0-9]+]]: index) {
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK:   %[[COL_LEN_I32:.+]] = tensor.extract %[[COLUMN_LENGTHS]][%[[ROW_IDX]]]
//       CHECK:   %[[COL_LEN:.+]] = arith.index_cast %[[COL_LEN_I32]] : i32 to index
//       CHECK:   %[[LINEARIZED:.+]] = affine.apply
//  CHECK-SAME:       affine_map<()[s0, s1] -> (s0 + s1)>()[%[[COL_LEN]], %[[COL_OFFSET]]]
//       CHECK:   "test.resolved_range"(%[[LINEARIZED]], %[[COL_SIZE]], %[[C1]],
//  CHECK-SAME:       %[[DENSE_OFFSET]], %[[DENSE_SIZE]], %[[C1]])

// -----

// Test failure when the outer sparse dimension size is not 1.
func.func public @outerSparseSizeNotOne(
    %source : memref<?x?xf32>, %column_lengths: memref<?xi32>,
    %num_rows: index, %row_idx: index, %col_offset: index,
    %col_size: index, %dense_offset: index, %dense_size: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %source_d0 = memref.dim %source, %c0 : memref<?x?xf32>
  %source_d1 = memref.dim %source, %c1 : memref<?x?xf32>
  // expected-error @+1 {{expected outer sparse dimension size and stride to be 1 in givenRanges}}
  %0 = iree_tensor_ext.cast_to_ragged_shape %source
      ragged_dim(0) column_lengths(%column_lengths) num_ragged_rows(%num_rows)
      : (memref<?x?xf32>{%source_d0, %source_d1}, memref<?xi32>)
      -> memref<?x?x?xf32, #iree_tensor_ext.ragged_shape<0>>
  "test.range"(%row_idx, %c2, %c1,
               %col_offset, %col_size, %c1,
               %dense_offset, %dense_size, %c1)
      : (index, index, index, index, index, index, index, index, index) -> ()
  return
}

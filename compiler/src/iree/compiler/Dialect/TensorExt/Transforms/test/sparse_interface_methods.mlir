// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-tensor-ext-test-sparse-op-interface-methods{test-lower-loop-range=true}, cse))' %s --split-input-file --allow-unregistered-dialect --mlir-print-local-scope --verify-diagnostics | FileCheck %s --check-prefixes=LOWER-LOOP-RANGE,CHECK-ALL
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-tensor-ext-test-sparse-op-interface-methods{test-get-estimated-loop-range=true}, cse))' %s --split-input-file --allow-unregistered-dialect --mlir-print-local-scope --verify-diagnostics | FileCheck %s --check-prefixes=GET-ESTIMATED-LOOP-RANGE,CHECK-ALL

// Check interface methods for a simple test case with a 2D ragged tensor.
func.func public @simpleTest(%source : memref<?x?xf32>, %column_lengths: memref<?xi32>,
    %num_rows: index, %lb0 : index, %lb1 : index, %step0 : index, %step1 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %source_d0 = memref.dim %source, %c0 : memref<?x?xf32>
  %source_d1 = memref.dim %source, %c1 : memref<?x?xf32>
  %0 = iree_tensor_ext.cast_to_ragged_shape %source
      ragged_dim(0) column_lengths(%column_lengths) num_ragged_rows(%num_rows)
      : (memref<?x?xf32>{%source_d0, %source_d1}, memref<?xi32>)
      -> memref<?x?x?xf32, #iree_tensor_ext.ragged_shape<0>>
  %d0 = memref.dim %0, %c0 : memref<?x?x?xf32, #iree_tensor_ext.ragged_shape<0>>
  %d1 = memref.dim %0, %c1 : memref<?x?x?xf32, #iree_tensor_ext.ragged_shape<0>>
  scf.forall (%i, %j) = (%lb0, %lb1) to (%d0, %d1) step (%step0, %step1) {
    "some_op"(%i, %j) : (index, index) -> ()
  } {iree_tensor_ext.sparse_iteration_dims = #iree_tensor_ext.sparse_iteration_dims<[0, 1]>}
  return
}
// CHECK-ALL-LABEL: func public @simpleTest(
//  CHECK-ALL-SAME:     %[[SOURCE:.+]]: memref<?x?xf32>,
//  CHECK-ALL-SAME:     %[[COLUMN_LENGTHS:.+]]: memref<?xi32>,
//  CHECK-ALL-SAME:     %[[NUM_ROWS:[a-zA-Z0-9]+]]: index,
//  CHECK-ALL-SAME:     %[[LB0:[a-zA-Z0-9]+]]: index
//  CHECK-ALL-SAME:     %[[LB1:[a-zA-Z0-9]+]]: index
//  CHECK-ALL-SAME:     %[[STEP0:[a-zA-Z0-9]+]]: index,
//  CHECK-ALL-SAME:     %[[STEP1:[a-zA-Z0-9]+]]: index) {

// LOWER-LOOP-RANGE:   %[[C1:.*]] = arith.constant 1 : index
// LOWER-LOOP-RANGE:   scf.for %[[IV0:.+]] = %[[LB0]] to %[[NUM_ROWS]] step %[[STEP0]] {
// LOWER-LOOP-RANGE:     %[[PLUSONE:.+]] = arith.addi %[[IV0]], %[[C1]]
// LOWER-LOOP-RANGE:     %[[COL_END_I32:.+]] = memref.load %[[COLUMN_LENGTHS]][%[[PLUSONE]]]
// LOWER-LOOP-RANGE:     %[[COL_START_I32:.+]] = memref.load %[[COLUMN_LENGTHS]][%[[IV0]]]
// LOWER-LOOP-RANGE:     %[[COL_END:.+]] = arith.index_cast %[[COL_END_I32]] : i32 to index
// LOWER-LOOP-RANGE:     %[[COL_START:.+]] = arith.index_cast %[[COL_START_I32]] : i32 to index
// LOWER-LOOP-RANGE:     %[[COL_RANGE:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%[[COL_END]], %[[COL_START]]]
// LOWER-LOOP-RANGE:     scf.for %[[IV1:.+]] = %[[LB1]] to %[[COL_RANGE]] step %[[STEP1]] {
// LOWER-LOOP-RANGE:       "some_op"(%[[IV0]], %[[IV1]])

// GET-ESTIMATED-LOOP-RANGE: %[[C0:.+]] = arith.constant 0 : index
// GET-ESTIMATED-LOOP-RANGE: %[[D0:.+]] = memref.dim %[[SOURCE]], %[[C0]]
// GET-ESTIMATED-LOOP-RANGE: %[[UB1:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 ceildiv s1)>()[%[[D0]], %[[NUM_ROWS]]]
// GET-ESTIMATED-LOOP-RANGE: scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]]) = (%[[LB0]], %[[LB1]]) to (%[[NUM_ROWS]], %[[UB1]]) step (%[[STEP0]], %[[STEP1]]) {
// GET-ESTIMATED-LOOP-RANGE:   "some_op"(%[[IV0]], %[[IV1]])

// -----

// Check interface methods for `cast_to_ragged_shape` with estimated column length.
func.func public @testEstimatedColumnLength(%source : memref<?x?xf32>, %column_lengths: memref<?xi32>,
    %num_rows: index, %lb0 : index, %lb1 : index,
    %step0 : index, %step1 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %source_d0 = memref.dim %source, %c0 : memref<?x?xf32>
  %source_d1 = memref.dim %source, %c1 : memref<?x?xf32>
  %0 = iree_tensor_ext.cast_to_ragged_shape %source
      ragged_dim(0) column_lengths(%column_lengths) num_ragged_rows(%num_rows)
      : (memref<?x?xf32>{%source_d0, %source_d1}, memref<?xi32>)
      -> memref<?x?x?xf32, #iree_tensor_ext.ragged_shape<0>>
  %d0 = memref.dim %0, %c0 : memref<?x?x?xf32, #iree_tensor_ext.ragged_shape<0>>
  %d1 = memref.dim %0, %c1 : memref<?x?x?xf32, #iree_tensor_ext.ragged_shape<0>>
  scf.forall (%i, %j) = (%lb0, %lb1) to (%d0, %d1) step (%step0, %step1) {
    "some_op"(%i, %j) : (index, index) -> ()
  } {iree_tensor_ext.sparse_iteration_dims = #iree_tensor_ext.sparse_iteration_dims<[0, 1]>}
  return
}
// CHECK-ALL-LABEL: func public @testEstimatedColumnLength(
//  CHECK-ALL-SAME:     %[[SOURCE:.+]]: memref<?x?xf32>,
//  CHECK-ALL-SAME:     %[[COLUMN_LENGTHS:.+]]: memref<?xi32>,
//  CHECK-ALL-SAME:     %[[NUM_ROWS:[a-zA-Z0-9]+]]: index,
//  CHECK-ALL-SAME:     %[[LB0:[a-zA-Z0-9]+]]: index
//  CHECK-ALL-SAME:     %[[LB1:[a-zA-Z0-9]+]]: index
//  CHECK-ALL-SAME:     %[[STEP0:[a-zA-Z0-9]+]]: index,
//  CHECK-ALL-SAME:     %[[STEP1:[a-zA-Z0-9]+]]: index) {

// LOWER-LOOP-RANGE:   %[[C1:.*]] = arith.constant 1 : index
// LOWER-LOOP-RANGE:   scf.for %[[IV0:.+]] = %[[LB0]] to %[[NUM_ROWS]] step %[[STEP0]] {
// LOWER-LOOP-RANGE:     %[[PLUSONE:.+]] = arith.addi %[[IV0]], %[[C1]]
// LOWER-LOOP-RANGE:     %[[COL_END_I32:.+]] = memref.load %[[COLUMN_LENGTHS]][%[[PLUSONE]]]
// LOWER-LOOP-RANGE:     %[[COL_START_I32:.+]] = memref.load %[[COLUMN_LENGTHS]][%[[IV0]]]
// LOWER-LOOP-RANGE:     %[[COL_END:.+]] = arith.index_cast %[[COL_END_I32]] : i32 to index
// LOWER-LOOP-RANGE:     %[[COL_START:.+]] = arith.index_cast %[[COL_START_I32]] : i32 to index
// LOWER-LOOP-RANGE:     %[[COL_RANGE:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%[[COL_END]], %[[COL_START]]]
// LOWER-LOOP-RANGE:     scf.for %[[IV1:.+]] = %[[LB1]] to %[[COL_RANGE]] step %[[STEP1]] {
// LOWER-LOOP-RANGE:       "some_op"(%[[IV0]], %[[IV1]])

//      GET-ESTIMATED-LOOP-RANGE: %[[EST_COLS:.+]] = affine.apply
// GET-ESTIMATED-LOOP-RANGE-SAME:     affine_map<()[s0, s1] -> (s0 ceildiv s1)>()[%[[SOURCE_D0:.+]], %[[NUM_ROWS]]]
//      GET-ESTIMATED-LOOP-RANGE: scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]]) = (%[[LB0]], %[[LB1]])
// GET-ESTIMATED-LOOP-RANGE-SAME:     to (%[[NUM_ROWS]], %[[EST_COLS]]) step (%[[STEP0]], %[[STEP1]]) {
//      GET-ESTIMATED-LOOP-RANGE:   "some_op"(%[[IV0]], %[[IV1]])

// -----

// Test case where the sparse loops are not the outer-most.

func.func public @nonOuterMostSparseLoops(%source : memref<?x?x?xf32>, %column_lengths: memref<?xi32>,
    %num_rows: index, %lb0 : index, %lb1 : index, %lb2 : index, %lb3 : index,
    %step0 : index, %step1 : index, %step2 : index, %step3 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %source_d0 = memref.dim %source, %c0 : memref<?x?x?xf32>
  %source_d1 = memref.dim %source, %c1 : memref<?x?x?xf32>
  %source_d2 = memref.dim %source, %c2 : memref<?x?x?xf32>
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
  } {iree_tensor_ext.sparse_iteration_dims = #iree_tensor_ext.sparse_iteration_dims<[1, 2]>}
  return
}
// CHECK-ALL-LABEL: func public @nonOuterMostSparseLoops(
//  CHECK-ALL-SAME:     %[[SOURCE:.+]]: memref<?x?x?xf32>,
//  CHECK-ALL-SAME:     %[[COLUMN_LENGTHS:.+]]: memref<?xi32>,
//  CHECK-ALL-SAME:     %[[NUM_ROWS:[a-zA-Z0-9]+]]: index,
//  CHECK-ALL-SAME:     %[[LB0:[a-zA-Z0-9]+]]: index,
//  CHECK-ALL-SAME:     %[[LB1:[a-zA-Z0-9]+]]: index,
//  CHECK-ALL-SAME:     %[[LB2:[a-zA-Z0-9]+]]: index,
//  CHECK-ALL-SAME:     %[[LB3:[a-zA-Z0-9]+]]: index,
//  CHECK-ALL-SAME:     %[[STEP0:[a-zA-Z0-9]+]]: index,
//  CHECK-ALL-SAME:     %[[STEP1:[a-zA-Z0-9]+]]: index,
//  CHECK-ALL-SAME:     %[[STEP2:[a-zA-Z0-9]+]]: index,
//  CHECK-ALL-SAME:     %[[STEP3:[a-zA-Z0-9]+]]: index) {

// LOWER-LOOP-RANGE:   %[[C1:.*]] = arith.constant 1 : index
// LOWER-LOOP-RANGE:   scf.for %[[IV0:.+]] =
// LOWER-LOOP-RANGE:     scf.for %[[IV1:.+]] = %[[LB1]] to %[[NUM_ROWS]] step %[[STEP1]] {
// LOWER-LOOP-RANGE:       %[[PLUSONE:.+]] = arith.addi %[[IV1]], %[[C1]]
// LOWER-LOOP-RANGE:       %[[COL_END_I32:.+]] = memref.load %[[COLUMN_LENGTHS]][%[[PLUSONE]]]
// LOWER-LOOP-RANGE:       %[[COL_START_I32:.+]] = memref.load %[[COLUMN_LENGTHS]][%[[IV1]]]
// LOWER-LOOP-RANGE:       %[[COL_END:.+]] = arith.index_cast %[[COL_END_I32]] : i32 to index
// LOWER-LOOP-RANGE:       %[[COL_START:.+]] = arith.index_cast %[[COL_START_I32]] : i32 to index
// LOWER-LOOP-RANGE:       %[[COL_RANGE:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%[[COL_END]], %[[COL_START]]]
// LOWER-LOOP-RANGE:       scf.for %[[IV2:.+]] = %[[LB2]] to %[[COL_RANGE]] step %[[STEP2]] {
// LOWER-LOOP-RANGE:         scf.for %[[IV3:.+]] =
// LOWER-LOOP-RANGE:           "some_op"(%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]])

//      GET-ESTIMATED-LOOP-RANGE: %[[C1:.+]] = arith.constant 1 : index
//      GET-ESTIMATED-LOOP-RANGE: %[[D1:.+]] = memref.dim %[[SOURCE]], %[[C1]]
//      GET-ESTIMATED-LOOP-RANGE: %[[UB2:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 ceildiv s1)>()[%[[D1]], %[[NUM_ROWS]]]
//      GET-ESTIMATED-LOOP-RANGE: scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]], %[[IV2:[a-zA-Z0-9]+]], %[[IV3:[a-zA-Z0-9]+]]
// GET-ESTIMATED-LOOP-RANGE-SAME:     = (%[[LB0]], %[[LB1]], %[[LB2]], %[[LB3]]) to (%{{[a-zA-Z0-9_]+}}, %[[NUM_ROWS]], %[[UB2]], %{{[a-zA-Z0-9_]+}})
// GET-ESTIMATED-LOOP-RANGE-SAME:     step (%[[STEP0]], %[[STEP1]], %[[STEP2]], %[[STEP3]]) {
//      GET-ESTIMATED-LOOP-RANGE:   "some_op"(%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]])

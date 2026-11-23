// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-tensor-ext-test-sparse-op-interface-methods{test-resolve-range-rewriters=true}))' %s --split-input-file --mlir-print-local-scope | FileCheck %s

// Test vector.transfer_read rewriting for cast_to_ragged_shape with ragged_dim(0)
// This tests a permutation map that drops the sparse dimensions.
func.func @test(%arg0 : index, %arg1: index, %arg2 : index, %num_ragged_rows : index,
    %source : memref<?x512xf16>, %column_lengths : memref<?xi32>, %d0 : index)
    -> vector<4xf16> {
  %cst = arith.constant 0.0 : f16
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(0)
      column_lengths(%column_lengths) num_ragged_rows(%num_ragged_rows)
      : (memref<?x512xf16>{%d0}, memref<?xi32>)
      -> memref<?x?x512xf16, #iree_tensor_ext.ragged_shape<0>>
  // Permutation map drops dimensions 0 and 1 (the sparse dimensions), only accesses dimension 2
  %1 = vector.transfer_read %0[%arg0, %arg1, %arg2], %cst
      {in_bounds = [true], permutation_map = affine_map<(d0, d1, d2) -> (d2)>}
      : memref<?x?x512xf16, #iree_tensor_ext.ragged_shape<0>>, vector<4xf16>
  return %1 : vector<4xf16>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[NUM_ROWS:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[SOURCE:[a-zA-Z0-9]+]]: memref<?x512xf16>
//  CHECK-SAME:     %[[COLUMN_LENGTHS:[a-zA-Z0-9]+]]: memref<?xi32>
//       CHECK:   %[[CST:.+]] = arith.constant 0.000000e+00 : f16
//       CHECK:   %[[COLUMN_LENGTH_I32:.+]] = memref.load %[[COLUMN_LENGTHS]][%[[ARG0]]]
//       CHECK:   %[[COLUMN_LENGTH:.+]] = arith.index_cast %[[COLUMN_LENGTH_I32]]
//       CHECK:   %[[LINEARIZED_OFFSET:.+]] = affine.apply
//  CHECK-SAME:       affine_map<()[s0, s1] -> (s0 + s1)>()[%[[COLUMN_LENGTH]], %[[ARG1]]]
//       CHECK:   %[[RESULT:.+]] = vector.transfer_read
//  CHECK-SAME:       %[[SOURCE]][%[[LINEARIZED_OFFSET]], %[[ARG2]]], %[[CST]]
//  CHECK-SAME:       {in_bounds = [true]} : memref<?x512xf16>, vector<4xf16>
//       CHECK:   return %[[RESULT]]

// -----

// Test vector.transfer_read rewriting with ragged_dim(1) - sparse dimensions at positions 1 and 2
// This tests a permutation map that drops the sparse dimensions which are NOT at the beginning.
func.func @test_ragged_dim_1(%arg0 : index, %arg1: index, %arg2 : index, %arg3 : index,
    %num_ragged_rows : index, %source : memref<?x?x512xf16>, %column_lengths : memref<?xi32>,
    %d0 : index, %d1: index) -> vector<4x4xf16> {
  %cst = arith.constant 0.0 : f16
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1)
      column_lengths(%column_lengths) num_ragged_rows(%num_ragged_rows)
      : (memref<?x?x512xf16>{%d0, %d1}, memref<?xi32>)
      -> memref<?x?x?x512xf16, #iree_tensor_ext.ragged_shape<1>>
  // Permutation map drops dimensions 1 and 2 (the sparse dimensions), accesses dimensions 0 and 3
  %1 = vector.transfer_read %0[%arg0, %arg1, %arg2, %arg3], %cst
      {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2, d3) -> (d0, d3)>}
      : memref<?x?x?x512xf16, #iree_tensor_ext.ragged_shape<1>>, vector<4x4xf16>
  return %1 : vector<4x4xf16>
}

// CHECK-LABEL: func @test_ragged_dim_1
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[NUM_ROWS:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[SOURCE:[a-zA-Z0-9]+]]: memref<?x?x512xf16>
//  CHECK-SAME:     %[[COLUMN_LENGTHS:[a-zA-Z0-9]+]]: memref<?xi32>
//       CHECK:   %[[CST:.+]] = arith.constant 0.000000e+00 : f16
//       CHECK:   %[[COLUMN_LENGTH_I32:.+]] = memref.load %[[COLUMN_LENGTHS]][%[[ARG1]]]
//       CHECK:   %[[COLUMN_LENGTH:.+]] = arith.index_cast %[[COLUMN_LENGTH_I32]]
//       CHECK:   %[[LINEARIZED_OFFSET:.+]] = affine.apply
//  CHECK-SAME:       affine_map<()[s0, s1] -> (s0 + s1)>()[%[[COLUMN_LENGTH]], %[[ARG2]]]
//       CHECK:   %[[RESULT:.+]] = vector.transfer_read
//  CHECK-SAME:       %[[SOURCE]][%[[ARG0]], %[[LINEARIZED_OFFSET]], %[[ARG3]]], %[[CST]]
//  CHECK-SAME:       {in_bounds = [true, true],
//  CHECK-SAME:        permutation_map = affine_map<(d0, d1, d2) -> (d0, d2)>}
//  CHECK-SAME:       : memref<?x?x512xf16>, vector<4x4xf16>
//       CHECK:   return %[[RESULT]]

// -----

// Test vector.load rewriting for cast_to_ragged_shape with ragged_dim(0)
// This tests loading a vector from the last dimension (non-sparse).
func.func @test_vector_load(%arg0 : index, %arg1: index, %arg2 : index, %num_ragged_rows : index,
    %source : memref<?x512xf16>, %column_lengths : memref<?xi32>, %d0 : index)
    -> vector<4xf16> {
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(0)
      column_lengths(%column_lengths) num_ragged_rows(%num_ragged_rows)
      : (memref<?x512xf16>{%d0}, memref<?xi32>)
      -> memref<?x?x512xf16, #iree_tensor_ext.ragged_shape<0>>
  // Load vector from the last dimension (non-sparse dimension 2)
  %1 = vector.load %0[%arg0, %arg1, %arg2] : memref<?x?x512xf16, #iree_tensor_ext.ragged_shape<0>>, vector<4xf16>
  return %1 : vector<4xf16>
}

// CHECK-LABEL: func @test_vector_load
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[NUM_ROWS:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[SOURCE:[a-zA-Z0-9]+]]: memref<?x512xf16>
//  CHECK-SAME:     %[[COLUMN_LENGTHS:[a-zA-Z0-9]+]]: memref<?xi32>
//       CHECK:   %[[COLUMN_LENGTH_I32:.+]] = memref.load %[[COLUMN_LENGTHS]][%[[ARG0]]]
//       CHECK:   %[[COLUMN_LENGTH:.+]] = arith.index_cast %[[COLUMN_LENGTH_I32]]
//       CHECK:   %[[LINEARIZED_OFFSET:.+]] = affine.apply
//  CHECK-SAME:       affine_map<()[s0, s1] -> (s0 + s1)>()[%[[COLUMN_LENGTH]], %[[ARG1]]]
//       CHECK:   %[[RESULT:.+]] = vector.load
//  CHECK-SAME:       %[[SOURCE]][%[[LINEARIZED_OFFSET]], %[[ARG2]]]
//  CHECK-SAME:       : memref<?x512xf16>, vector<4xf16>
//       CHECK:   return %[[RESULT]]

// -----

// Test vector.load rewriting with ragged_dim(1) - sparse dimensions at positions 1 and 2
// This tests loading a vector from the last dimension (non-sparse).
func.func @test_vector_load_ragged_dim_1(%arg0 : index, %arg1: index, %arg2 : index, %arg3 : index,
    %num_ragged_rows : index, %source : memref<?x?x512xf16>, %column_lengths : memref<?xi32>,
    %d0 : index, %d1: index) -> vector<4xf16> {
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1)
      column_lengths(%column_lengths) num_ragged_rows(%num_ragged_rows)
      : (memref<?x?x512xf16>{%d0, %d1}, memref<?xi32>)
      -> memref<?x?x?x512xf16, #iree_tensor_ext.ragged_shape<1>>
  // Load vector from the last dimension (non-sparse dimension 3)
  %1 = vector.load %0[%arg0, %arg1, %arg2, %arg3] : memref<?x?x?x512xf16, #iree_tensor_ext.ragged_shape<1>>, vector<4xf16>
  return %1 : vector<4xf16>
}

// CHECK-LABEL: func @test_vector_load_ragged_dim_1
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[NUM_ROWS:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[SOURCE:[a-zA-Z0-9]+]]: memref<?x?x512xf16>
//  CHECK-SAME:     %[[COLUMN_LENGTHS:[a-zA-Z0-9]+]]: memref<?xi32>
//       CHECK:   %[[COLUMN_LENGTH_I32:.+]] = memref.load %[[COLUMN_LENGTHS]][%[[ARG1]]]
//       CHECK:   %[[COLUMN_LENGTH:.+]] = arith.index_cast %[[COLUMN_LENGTH_I32]]
//       CHECK:   %[[LINEARIZED_OFFSET:.+]] = affine.apply
//  CHECK-SAME:       affine_map<()[s0, s1] -> (s0 + s1)>()[%[[COLUMN_LENGTH]], %[[ARG2]]]
//       CHECK:   %[[RESULT:.+]] = vector.load
//  CHECK-SAME:       %[[SOURCE]][%[[ARG0]], %[[LINEARIZED_OFFSET]], %[[ARG3]]]
//  CHECK-SAME:       : memref<?x?x512xf16>, vector<4xf16>
//       CHECK:   return %[[RESULT]]

// -----

// Test vector.maskedload rewriting for cast_to_ragged_shape with ragged_dim(0)
// This tests loading a masked vector from the last dimension (non-sparse).
func.func @test_vector_maskedload(%arg0 : index, %arg1: index, %arg2 : index, %num_ragged_rows : index,
    %source : memref<?x512xf16>, %column_lengths : memref<?xi32>, %d0 : index,
    %mask : vector<4xi1>, %pass_thru : vector<4xf16>)
    -> vector<4xf16> {
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(0)
      column_lengths(%column_lengths) num_ragged_rows(%num_ragged_rows)
      : (memref<?x512xf16>{%d0}, memref<?xi32>)
      -> memref<?x?x512xf16, #iree_tensor_ext.ragged_shape<0>>
  // Load masked vector from the last dimension (non-sparse dimension 2)
  %1 = vector.maskedload %0[%arg0, %arg1, %arg2], %mask, %pass_thru : memref<?x?x512xf16, #iree_tensor_ext.ragged_shape<0>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
  return %1 : vector<4xf16>
}

// CHECK-LABEL: func @test_vector_maskedload
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[NUM_ROWS:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[SOURCE:[a-zA-Z0-9]+]]: memref<?x512xf16>
//  CHECK-SAME:     %[[COLUMN_LENGTHS:[a-zA-Z0-9]+]]: memref<?xi32>
//  CHECK-SAME:     %[[D0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[MASK:[a-zA-Z0-9]+]]: vector<4xi1>
//  CHECK-SAME:     %[[PASS_THRU:[a-zA-Z0-9]+]]: vector<4xf16>
//       CHECK:   %[[COLUMN_LENGTH_I32:.+]] = memref.load %[[COLUMN_LENGTHS]][%[[ARG0]]]
//       CHECK:   %[[COLUMN_LENGTH:.+]] = arith.index_cast %[[COLUMN_LENGTH_I32]]
//       CHECK:   %[[LINEARIZED_OFFSET:.+]] = affine.apply
//  CHECK-SAME:       affine_map<()[s0, s1] -> (s0 + s1)>()[%[[COLUMN_LENGTH]], %[[ARG1]]]
//       CHECK:   %[[RESULT:.+]] = vector.maskedload
//  CHECK-SAME:       %[[SOURCE]][%[[LINEARIZED_OFFSET]], %[[ARG2]]], %[[MASK]], %[[PASS_THRU]]
//  CHECK-SAME:       : memref<?x512xf16>, vector<4xi1>, vector<4xf16> into vector<4xf16>
//       CHECK:   return %[[RESULT]]

// -----

// Test vector.maskedload rewriting with ragged_dim(1) - sparse dimensions at positions 1 and 2
// This tests loading a masked vector from the last dimension (non-sparse).
func.func @test_vector_maskedload_ragged_dim_1(%arg0 : index, %arg1: index, %arg2 : index, %arg3 : index,
    %num_ragged_rows : index, %source : memref<?x?x512xf16>, %column_lengths : memref<?xi32>,
    %d0 : index, %d1: index, %mask : vector<4xi1>, %pass_thru : vector<4xf16>) -> vector<4xf16> {
  %0 = iree_tensor_ext.cast_to_ragged_shape %source ragged_dim(1)
      column_lengths(%column_lengths) num_ragged_rows(%num_ragged_rows)
      : (memref<?x?x512xf16>{%d0, %d1}, memref<?xi32>)
      -> memref<?x?x?x512xf16, #iree_tensor_ext.ragged_shape<1>>
  // Load masked vector from the last dimension (non-sparse dimension 3)
  %1 = vector.maskedload %0[%arg0, %arg1, %arg2, %arg3], %mask, %pass_thru : memref<?x?x?x512xf16, #iree_tensor_ext.ragged_shape<1>>, vector<4xi1>, vector<4xf16> into vector<4xf16>
  return %1 : vector<4xf16>
}

// CHECK-LABEL: func @test_vector_maskedload_ragged_dim_1
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[NUM_ROWS:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[SOURCE:[a-zA-Z0-9]+]]: memref<?x?x512xf16>
//  CHECK-SAME:     %[[COLUMN_LENGTHS:[a-zA-Z0-9]+]]: memref<?xi32>
//  CHECK-SAME:     %[[D0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[D1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[MASK:[a-zA-Z0-9]+]]: vector<4xi1>
//  CHECK-SAME:     %[[PASS_THRU:[a-zA-Z0-9]+]]: vector<4xf16>
//       CHECK:   %[[COLUMN_LENGTH_I32:.+]] = memref.load %[[COLUMN_LENGTHS]][%[[ARG1]]]
//       CHECK:   %[[COLUMN_LENGTH:.+]] = arith.index_cast %[[COLUMN_LENGTH_I32]]
//       CHECK:   %[[LINEARIZED_OFFSET:.+]] = affine.apply
//  CHECK-SAME:       affine_map<()[s0, s1] -> (s0 + s1)>()[%[[COLUMN_LENGTH]], %[[ARG2]]]
//       CHECK:   %[[RESULT:.+]] = vector.maskedload
//  CHECK-SAME:       %[[SOURCE]][%[[ARG0]], %[[LINEARIZED_OFFSET]], %[[ARG3]]], %[[MASK]], %[[PASS_THRU]]
//  CHECK-SAME:       : memref<?x?x512xf16>, vector<4xi1>, vector<4xf16> into vector<4xf16>
//       CHECK:   return %[[RESULT]]

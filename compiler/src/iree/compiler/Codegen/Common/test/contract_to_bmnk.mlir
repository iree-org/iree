// RUN: iree-opt --pass-pipeline="builtin.module(iree-codegen-test-contract-to-bmnk-patterns)" --split-input-file %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Batch dim tests
//===----------------------------------------------------------------------===//

// Batch dim 'b' is not outermost in LHS/RHS: maps have b at inner positions.
// Batch dim is already at iteration position 0; only operand transpositions needed.

//      CHECK-DAG: #[[$MAP_LHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//      CHECK-DAG: #[[$MAP_RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//      CHECK-DAG: #[[$MAP_ACC:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL: func @batch_dim_operand_transpose
// CHECK-SAME: %[[A:.+]]: vector<4x2x3xf32>,
// CHECK-SAME: %[[B:.+]]: vector<3x2x5xf32>,
// CHECK-SAME: %[[C:.+]]: vector<2x4x5xf32>
func.func @batch_dim_operand_transpose(
    %A: vector<4x2x3xf32>,
    %B: vector<3x2x5xf32>,
    %C: vector<2x4x5xf32>) -> vector<2x4x5xf32> {

  // CHECK-DAG: %[[AT:.+]] = vector.transpose %[[A]], [1, 0, 2] : vector<4x2x3xf32> to vector<2x4x3xf32>
  // CHECK-DAG: %[[BT:.+]] = vector.transpose %[[B]], [1, 2, 0] : vector<3x2x5xf32> to vector<2x5x3xf32>
  // CHECK: vector.contract
  // CHECK-SAME: indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_ACC]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  // CHECK-SAME: %[[AT]], %[[BT]], %[[C]]
  // CHECK-SAME: : vector<2x4x3xf32>, vector<2x5x3xf32> into vector<2x4x5xf32>

  %result = vector.contract {
      indexing_maps = [
          affine_map<(b, m, n, k) -> (m, b, k)>,
          affine_map<(b, m, n, k) -> (k, b, n)>,
          affine_map<(b, m, n, k) -> (b, m, n)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<4x2x3xf32>, vector<3x2x5xf32> into vector<2x4x5xf32>

  return %result : vector<2x4x5xf32>
}

// -----

// Batch dim at non-zero iteration-space position.
// Iteration space permutation needed, plus operand and result transposes.

//      CHECK-DAG: #[[$MAP_LHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//      CHECK-DAG: #[[$MAP_RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//      CHECK-DAG: #[[$MAP_ACC:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL: func @batch_dim_iter_space
// CHECK-SAME: %[[A:.+]]: vector<2x4x3xf32>,
// CHECK-SAME: %[[B:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[C:.+]]: vector<4x2x5xf32>
func.func @batch_dim_iter_space(
    %A: vector<2x4x3xf32>,
    %B: vector<2x3x5xf32>,
    %C: vector<4x2x5xf32>) -> vector<4x2x5xf32> {

  // CHECK-DAG: %[[BT:.+]] = vector.transpose %[[B]], [0, 2, 1] : vector<2x3x5xf32> to vector<2x5x3xf32>
  // CHECK-DAG: %[[CT:.+]] = vector.transpose %[[C]], [1, 0, 2] : vector<4x2x5xf32> to vector<2x4x5xf32>
  // CHECK: %[[RES:.+]] = vector.contract
  // CHECK-SAME: indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_ACC]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  // CHECK-SAME: %[[A]], %[[BT]], %[[CT]]
  // CHECK-SAME: : vector<2x4x3xf32>, vector<2x5x3xf32> into vector<2x4x5xf32>
  // CHECK: vector.transpose %[[RES]], [1, 0, 2] : vector<2x4x5xf32> to vector<4x2x5xf32>

  %result = vector.contract {
      indexing_maps = [
          affine_map<(m, b, n, k) -> (b, m, k)>,
          affine_map<(m, b, n, k) -> (b, k, n)>,
          affine_map<(m, b, n, k) -> (m, b, n)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<2x4x3xf32>, vector<2x3x5xf32> into vector<4x2x5xf32>

  return %result : vector<4x2x5xf32>
}

// -----

// Masked: batch dim at non-zero iteration-space position.

//      CHECK-DAG: #[[$MAP_LHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//      CHECK-DAG: #[[$MAP_RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//      CHECK-DAG: #[[$MAP_ACC:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL: func @batch_dim_masked
// CHECK-SAME: %[[A:.+]]: vector<2x4x3xf32>,
// CHECK-SAME: %[[B:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[C:.+]]: vector<4x2x5xf32>,
// CHECK-SAME: %[[MASK:.+]]: vector<4x2x5x3xi1>
func.func @batch_dim_masked(
    %A: vector<2x4x3xf32>,
    %B: vector<2x3x5xf32>,
    %C: vector<4x2x5xf32>,
    %mask: vector<4x2x5x3xi1>) -> vector<4x2x5xf32> {

  // CHECK-DAG: %[[BT:.+]] = vector.transpose %[[B]], [0, 2, 1] : vector<2x3x5xf32> to vector<2x5x3xf32>
  // CHECK-DAG: %[[CT:.+]] = vector.transpose %[[C]], [1, 0, 2] : vector<4x2x5xf32> to vector<2x4x5xf32>
  // CHECK-DAG: %[[MT:.+]] = vector.transpose %[[MASK]], [1, 0, 2, 3] : vector<4x2x5x3xi1> to vector<2x4x5x3xi1>
  // CHECK: %[[RES:.+]] = vector.mask %[[MT]] {
  // CHECK:   vector.contract
  // CHECK-SAME: indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_ACC]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  // CHECK-SAME: %[[A]], %[[BT]], %[[CT]]
  // CHECK-SAME: : vector<2x4x3xf32>, vector<2x5x3xf32> into vector<2x4x5xf32>
  // CHECK: } : vector<2x4x5x3xi1> -> vector<2x4x5xf32>
  // CHECK: vector.transpose %[[RES]], [1, 0, 2] : vector<2x4x5xf32> to vector<4x2x5xf32>

  %result = vector.mask %mask {
    vector.contract {
        indexing_maps = [
            affine_map<(m, b, n, k) -> (b, m, k)>,
            affine_map<(m, b, n, k) -> (b, k, n)>,
            affine_map<(m, b, n, k) -> (m, b, n)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } %A, %B, %C : vector<2x4x3xf32>, vector<2x3x5xf32> into vector<4x2x5xf32>
  } : vector<4x2x5x3xi1> -> vector<4x2x5xf32>

  return %result : vector<4x2x5xf32>
}

// -----

// Two batch dims: b1 is already at position 0 (canonical), but b2 is at
// iteration position 3 and operand position 2, not at position 1.

//      CHECK-DAG: #[[$MAP_LHS:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
//      CHECK-DAG: #[[$MAP_RHS:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
//      CHECK-DAG: #[[$MAP_ACC:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @two_batch_dims_second_not_canonical
// CHECK-SAME: %[[A:.+]]: vector<2x4x3x6xf32>,
// CHECK-SAME: %[[B:.+]]: vector<2x3x6x5xf32>,
// CHECK-SAME: %[[C:.+]]: vector<2x4x3x5xf32>
func.func @two_batch_dims_second_not_canonical(
    %A: vector<2x4x3x6xf32>,
    %B: vector<2x3x6x5xf32>,
    %C: vector<2x4x3x5xf32>) -> vector<2x4x3x5xf32> {

  // CHECK-DAG: %[[AT:.+]] = vector.transpose %[[A]], [0, 2, 1, 3] : vector<2x4x3x6xf32> to vector<2x3x4x6xf32>
  // CHECK-DAG: %[[BT:.+]] = vector.transpose %[[B]], [0, 1, 3, 2] : vector<2x3x6x5xf32> to vector<2x3x5x6xf32>
  // CHECK-DAG: %[[CT:.+]] = vector.transpose %[[C]], [0, 2, 1, 3] : vector<2x4x3x5xf32> to vector<2x3x4x5xf32>
  // CHECK: %[[RES:.+]] = vector.contract
  // CHECK-SAME: indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_ACC]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
  // CHECK-SAME: %[[AT]], %[[BT]], %[[CT]]
  // CHECK-SAME: : vector<2x3x4x6xf32>, vector<2x3x5x6xf32> into vector<2x3x4x5xf32>
  // CHECK: vector.transpose %[[RES]], [0, 2, 1, 3] : vector<2x3x4x5xf32> to vector<2x4x3x5xf32>

  %result = vector.contract {
      indexing_maps = [
          affine_map<(b1, m, n, b2, k) -> (b1, m, b2, k)>,
          affine_map<(b1, m, n, b2, k) -> (b1, b2, k, n)>,
          affine_map<(b1, m, n, b2, k) -> (b1, m, b2, n)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<2x4x3x6xf32>, vector<2x3x6x5xf32> into vector<2x4x3x5xf32>

  return %result : vector<2x4x3x5xf32>
}

// -----

// Batch dim already at canonical iteration position, but RHS free dim
// isn't in canonical operand position within RHS. Only RHS transpose needed.

//      CHECK-DAG: #[[$MAP_LHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//      CHECK-DAG: #[[$MAP_RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//      CHECK-DAG: #[[$MAP_ACC:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL: func @batch_rhs_not_canonical
// CHECK-SAME: %[[A:.+]]: vector<2x4x3xf32>,
// CHECK-SAME: %[[B:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[C:.+]]: vector<2x4x5xf32>
func.func @batch_rhs_not_canonical(
    %A: vector<2x4x3xf32>,
    %B: vector<2x3x5xf32>,
    %C: vector<2x4x5xf32>) -> vector<2x4x5xf32> {

  // CHECK: %[[BT:.+]] = vector.transpose %[[B]], [0, 2, 1] : vector<2x3x5xf32> to vector<2x5x3xf32>
  // CHECK: vector.contract
  // CHECK-SAME: indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_ACC]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  // CHECK-SAME: %[[A]], %[[BT]], %[[C]]
  // CHECK-SAME: : vector<2x4x3xf32>, vector<2x5x3xf32> into vector<2x4x5xf32>

  %result = vector.contract {
      indexing_maps = [
          affine_map<(b, m, n, k) -> (b, m, k)>,
          affine_map<(b, m, n, k) -> (b, k, n)>,
          affine_map<(b, m, n, k) -> (b, m, n)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<2x4x3xf32>, vector<2x3x5xf32> into vector<2x4x5xf32>

  return %result : vector<2x4x5xf32>
}

// -----

//===----------------------------------------------------------------------===//
// LHS free dim tests
//===----------------------------------------------------------------------===//

// LHS free dim 'm' is not outermost in LHS: map is (k, m) instead of (m, k).
// RHS free dim 'n' is not outermost in RHS: map is (k, n) instead of (n, k).

//      CHECK-DAG: #[[$MAP_LHS:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//      CHECK-DAG: #[[$MAP_RHS:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//      CHECK-DAG: #[[$MAP_ACC:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func @lhs_free_dim_operand_transpose
// CHECK-SAME: %[[A:.+]]: vector<8x4xf32>,
// CHECK-SAME: %[[B:.+]]: vector<8x6xf32>,
// CHECK-SAME: %[[C:.+]]: vector<4x6xf32>
func.func @lhs_free_dim_operand_transpose(
    %A: vector<8x4xf32>,
    %B: vector<8x6xf32>,
    %C: vector<4x6xf32>) -> vector<4x6xf32> {

  // CHECK-DAG: %[[AT:.+]] = vector.transpose %[[A]], [1, 0] : vector<8x4xf32> to vector<4x8xf32>
  // CHECK-DAG: %[[BT:.+]] = vector.transpose %[[B]], [1, 0] : vector<8x6xf32> to vector<6x8xf32>
  // CHECK: vector.contract
  // CHECK-SAME: indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_ACC]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK-SAME: %[[AT]], %[[BT]], %[[C]]
  // CHECK-SAME: : vector<4x8xf32>, vector<6x8xf32> into vector<4x6xf32>

  %result = vector.contract {
      indexing_maps = [
          affine_map<(m, n, k) -> (k, m)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<8x4xf32>, vector<8x6xf32> into vector<4x6xf32>

  return %result : vector<4x6xf32>
}

// -----

// LHS free dim 'm' is at iteration-space position 1 (after n).
// Needs iteration-space reordering + ACC transpose.

//      CHECK-DAG: #[[$MAP_LHS:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//      CHECK-DAG: #[[$MAP_RHS:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//      CHECK-DAG: #[[$MAP_ACC:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func @lhs_free_dim_iter_space
// CHECK-SAME: %[[A:.+]]: vector<4x8xf32>,
// CHECK-SAME: %[[B:.+]]: vector<6x8xf32>,
// CHECK-SAME: %[[C:.+]]: vector<6x4xf32>
func.func @lhs_free_dim_iter_space(
    %A: vector<4x8xf32>,
    %B: vector<6x8xf32>,
    %C: vector<6x4xf32>) -> vector<6x4xf32> {

  // ACC needs transpose (n,m) -> (m,n), result transposed back.
  // CHECK-DAG: %[[CT:.+]] = vector.transpose %[[C]], [1, 0] : vector<6x4xf32> to vector<4x6xf32>
  // CHECK: %[[RES:.+]] = vector.contract
  // CHECK-SAME: indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_ACC]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK-SAME: %[[A]], %[[B]], %[[CT]]
  // CHECK-SAME: : vector<4x8xf32>, vector<6x8xf32> into vector<4x6xf32>
  // CHECK: vector.transpose %[[RES]], [1, 0] : vector<4x6xf32> to vector<6x4xf32>

  %result = vector.contract {
      indexing_maps = [
          affine_map<(n, m, k) -> (m, k)>,
          affine_map<(n, m, k) -> (n, k)>,
          affine_map<(n, m, k) -> (n, m)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<4x8xf32>, vector<6x8xf32> into vector<6x4xf32>

  return %result : vector<6x4xf32>
}

// -----

// Masked: LHS free dim 'm' at iteration-space position 1.
// Mask must be transposed along with the iteration space.

//      CHECK-DAG: #[[$MAP_LHS:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//      CHECK-DAG: #[[$MAP_RHS:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//      CHECK-DAG: #[[$MAP_ACC:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func @lhs_free_dim_masked
// CHECK-SAME: %[[A:.+]]: vector<4x8xf32>,
// CHECK-SAME: %[[B:.+]]: vector<6x8xf32>,
// CHECK-SAME: %[[C:.+]]: vector<6x4xf32>,
// CHECK-SAME: %[[MASK:.+]]: vector<6x4x8xi1>
func.func @lhs_free_dim_masked(
    %A: vector<4x8xf32>,
    %B: vector<6x8xf32>,
    %C: vector<6x4xf32>,
    %mask: vector<6x4x8xi1>) -> vector<6x4xf32> {

  // CHECK-DAG: %[[CT:.+]] = vector.transpose %[[C]], [1, 0] : vector<6x4xf32> to vector<4x6xf32>
  // CHECK-DAG: %[[MT:.+]] = vector.transpose %[[MASK]], [1, 0, 2] : vector<6x4x8xi1> to vector<4x6x8xi1>
  // CHECK: %[[RES:.+]] = vector.mask %[[MT]] {
  // CHECK:   vector.contract
  // CHECK-SAME: indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_ACC]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK-SAME: %[[A]], %[[B]], %[[CT]]
  // CHECK-SAME: : vector<4x8xf32>, vector<6x8xf32> into vector<4x6xf32>
  // CHECK: } : vector<4x6x8xi1> -> vector<4x6xf32>
  // CHECK: vector.transpose %[[RES]], [1, 0] : vector<4x6xf32> to vector<6x4xf32>

  %result = vector.mask %mask {
    vector.contract {
        indexing_maps = [
            affine_map<(n, m, k) -> (m, k)>,
            affine_map<(n, m, k) -> (n, k)>,
            affine_map<(n, m, k) -> (n, m)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
    } %A, %B, %C : vector<4x8xf32>, vector<6x8xf32> into vector<6x4xf32>
  } : vector<6x4x8xi1> -> vector<6x4xf32>

  return %result : vector<6x4xf32>
}

// -----

// Two LHS-free dims: m1 already at position 0 (canonical), but m2 is at
// iteration position 2 and ACC position 2, not at position 1.

//      CHECK-DAG: #[[$MAP_LHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//      CHECK-DAG: #[[$MAP_RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
//      CHECK-DAG: #[[$MAP_ACC:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL: func @two_lhs_free_dims_second_not_canonical
// CHECK-SAME: %[[A:.+]]: vector<4x3x6xf32>,
// CHECK-SAME: %[[B:.+]]: vector<5x6xf32>,
// CHECK-SAME: %[[C:.+]]: vector<4x5x3xf32>
func.func @two_lhs_free_dims_second_not_canonical(
    %A: vector<4x3x6xf32>,
    %B: vector<5x6xf32>,
    %C: vector<4x5x3xf32>) -> vector<4x5x3xf32> {

  // m2 needs to move from ACC position 2 to 1.
  // m2 needs to move from iteration position 2 to 1.
  // CHECK-DAG: %[[CT:.+]] = vector.transpose %[[C]], [0, 2, 1] : vector<4x5x3xf32> to vector<4x3x5xf32>
  // CHECK: %[[RES:.+]] = vector.contract
  // CHECK-SAME: indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_ACC]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  // CHECK-SAME: %[[A]], %[[B]], %[[CT]]
  // CHECK-SAME: : vector<4x3x6xf32>, vector<5x6xf32> into vector<4x3x5xf32>
  // CHECK: vector.transpose %[[RES]], [0, 2, 1] : vector<4x3x5xf32> to vector<4x5x3xf32>

  %result = vector.contract {
      indexing_maps = [
          affine_map<(m1, n, m2, k) -> (m1, m2, k)>,
          affine_map<(m1, n, m2, k) -> (n, k)>,
          affine_map<(m1, n, m2, k) -> (m1, n, m2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<4x3x6xf32>, vector<5x6xf32> into vector<4x5x3xf32>

  return %result : vector<4x5x3xf32>
}

// -----

// LHS free dim at canonical iteration position, but RHS free dim isn't
// in canonical operand position within RHS. Only RHS transpose needed.

//      CHECK-DAG: #[[$MAP_LHS:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//      CHECK-DAG: #[[$MAP_RHS:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//      CHECK-DAG: #[[$MAP_ACC:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func @lhs_canonical_rhs_not
// CHECK-SAME: %[[A:.+]]: vector<4x8xf32>,
// CHECK-SAME: %[[B:.+]]: vector<8x6xf32>,
// CHECK-SAME: %[[C:.+]]: vector<4x6xf32>
func.func @lhs_canonical_rhs_not(
    %A: vector<4x8xf32>,
    %B: vector<8x6xf32>,
    %C: vector<4x6xf32>) -> vector<4x6xf32> {

  // CHECK: %[[BT:.+]] = vector.transpose %[[B]], [1, 0] : vector<8x6xf32> to vector<6x8xf32>
  // CHECK: vector.contract
  // CHECK-SAME: indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_ACC]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK-SAME: %[[A]], %[[BT]], %[[C]]
  // CHECK-SAME: : vector<4x8xf32>, vector<6x8xf32> into vector<4x6xf32>

  %result = vector.contract {
      indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<4x8xf32>, vector<8x6xf32> into vector<4x6xf32>

  return %result : vector<4x6xf32>
}

// -----

//===----------------------------------------------------------------------===//
// RHS free dim tests
//===----------------------------------------------------------------------===//

// RHS free dim 'n' is not outermost in RHS: map is (k, n) instead of (n, k).
// Iteration space needs reordering: m (lhs-free) should come before n (rhs-free).

//      CHECK-DAG: #[[$MAP_LHS:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//      CHECK-DAG: #[[$MAP_RHS:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//      CHECK-DAG: #[[$MAP_ACC:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func @rhs_free_dim_operand_transpose
// CHECK-SAME: %[[A:.+]]: vector<4x8xf32>,
// CHECK-SAME: %[[B:.+]]: vector<8x6xf32>,
// CHECK-SAME: %[[C:.+]]: vector<6x4xf32>
func.func @rhs_free_dim_operand_transpose(
    %A: vector<4x8xf32>,
    %B: vector<8x6xf32>,
    %C: vector<6x4xf32>) -> vector<6x4xf32> {

  // CHECK-DAG: %[[BT:.+]] = vector.transpose %[[B]], [1, 0] : vector<8x6xf32> to vector<6x8xf32>
  // CHECK-DAG: %[[CT:.+]] = vector.transpose %[[C]], [1, 0] : vector<6x4xf32> to vector<4x6xf32>
  // CHECK: %[[RES:.+]] = vector.contract
  // CHECK-SAME: indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_ACC]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK-SAME: %[[A]], %[[BT]], %[[CT]]
  // CHECK-SAME: : vector<4x8xf32>, vector<6x8xf32> into vector<4x6xf32>
  // CHECK: vector.transpose %[[RES]], [1, 0] : vector<4x6xf32> to vector<6x4xf32>

  %result = vector.contract {
      indexing_maps = [
          affine_map<(n, m, k) -> (m, k)>,
          affine_map<(n, m, k) -> (k, n)>,
          affine_map<(n, m, k) -> (n, m)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<4x8xf32>, vector<8x6xf32> into vector<6x4xf32>

  return %result : vector<6x4xf32>
}

// -----

// Two RHS-free dims: n1 already at position 0, but m (lhs-free) needs to
// move before both n1 and n2 in the unified canonical layout.

//      CHECK-DAG: #[[$MAP_LHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
//      CHECK-DAG: #[[$MAP_RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
//      CHECK-DAG: #[[$MAP_ACC:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL: func @two_rhs_free_dims_second_not_canonical
// CHECK-SAME: %[[A:.+]]: vector<4x6xf32>,
// CHECK-SAME: %[[B:.+]]: vector<5x3x6xf32>,
// CHECK-SAME: %[[C:.+]]: vector<5x4x3xf32>
func.func @two_rhs_free_dims_second_not_canonical(
    %A: vector<4x6xf32>,
    %B: vector<5x3x6xf32>,
    %C: vector<5x4x3xf32>) -> vector<5x4x3xf32> {

  // CHECK-DAG: %[[CT:.+]] = vector.transpose %[[C]], [1, 0, 2] : vector<5x4x3xf32> to vector<4x5x3xf32>
  // CHECK: %[[RES:.+]] = vector.contract
  // CHECK-SAME: indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_ACC]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  // CHECK-SAME: %[[A]], %[[B]], %[[CT]]
  // CHECK-SAME: : vector<4x6xf32>, vector<5x3x6xf32> into vector<4x5x3xf32>
  // CHECK: vector.transpose %[[RES]], [1, 0, 2] : vector<4x5x3xf32> to vector<5x4x3xf32>

  %result = vector.contract {
      indexing_maps = [
          affine_map<(n1, m, n2, k) -> (m, k)>,
          affine_map<(n1, m, n2, k) -> (n1, n2, k)>,
          affine_map<(n1, m, n2, k) -> (n1, m, n2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<4x6xf32>, vector<5x3x6xf32> into vector<5x4x3xf32>

  return %result : vector<5x4x3xf32>
}

// -----

// RHS free dim is canonical within RHS operand, but iteration space has
// n (rhs-free) before m (lhs-free). Unified layout requires m before n.

//      CHECK-DAG: #[[$MAP_LHS:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//      CHECK-DAG: #[[$MAP_RHS:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//      CHECK-DAG: #[[$MAP_ACC:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func @rhs_iter_not_canonical
// CHECK-SAME: %[[A:.+]]: vector<4x8xf32>,
// CHECK-SAME: %[[B:.+]]: vector<6x8xf32>,
// CHECK-SAME: %[[C:.+]]: vector<6x4xf32>
func.func @rhs_iter_not_canonical(
    %A: vector<4x8xf32>,
    %B: vector<6x8xf32>,
    %C: vector<6x4xf32>) -> vector<6x4xf32> {

  // CHECK-DAG: %[[CT:.+]] = vector.transpose %[[C]], [1, 0] : vector<6x4xf32> to vector<4x6xf32>
  // CHECK: %[[RES:.+]] = vector.contract
  // CHECK-SAME: indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_ACC]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK-SAME: %[[A]], %[[B]], %[[CT]]
  // CHECK-SAME: : vector<4x8xf32>, vector<6x8xf32> into vector<4x6xf32>
  // CHECK: vector.transpose %[[RES]], [1, 0] : vector<4x6xf32> to vector<6x4xf32>

  %result = vector.contract {
      indexing_maps = [
          affine_map<(n, m, k) -> (m, k)>,
          affine_map<(n, m, k) -> (n, k)>,
          affine_map<(n, m, k) -> (n, m)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<4x8xf32>, vector<6x8xf32> into vector<6x4xf32>

  return %result : vector<6x4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Negative tests (already in unified canonical form)
//===----------------------------------------------------------------------===//

// Already canonical: (m, n, k) with LHS=(m,k), RHS=(n,k), ACC=(m,n).

//      CHECK-DAG: #[[$MAP_LHS:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//      CHECK-DAG: #[[$MAP_RHS:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//      CHECK-DAG: #[[$MAP_ACC:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func @already_canonical
func.func @already_canonical(
    %A: vector<4x8xf32>,
    %B: vector<6x8xf32>,
    %C: vector<4x6xf32>) -> vector<4x6xf32> {

  // CHECK-NOT: vector.transpose
  // CHECK: vector.contract
  // CHECK-SAME: indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_ACC]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]

  %result = vector.contract {
      indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (n, k)>,
          affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<4x8xf32>, vector<6x8xf32> into vector<4x6xf32>

  return %result : vector<4x6xf32>
}

// -----

// Already canonical with batch: (b, m, n, k) with
// LHS=(b,m,k), RHS=(b,n,k), ACC=(b,m,n).

//      CHECK-DAG: #[[$MAP_LHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//      CHECK-DAG: #[[$MAP_RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//      CHECK-DAG: #[[$MAP_ACC:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL: func @already_canonical_with_batch
func.func @already_canonical_with_batch(
    %A: vector<2x4x3xf32>,
    %B: vector<2x5x3xf32>,
    %C: vector<2x4x5xf32>) -> vector<2x4x5xf32> {

  // CHECK-NOT: vector.transpose
  // CHECK: vector.contract
  // CHECK-SAME: indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_ACC]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]

  %result = vector.contract {
      indexing_maps = [
          affine_map<(b, m, n, k) -> (b, m, k)>,
          affine_map<(b, m, n, k) -> (b, n, k)>,
          affine_map<(b, m, n, k) -> (b, m, n)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<2x4x3xf32>, vector<2x5x3xf32> into vector<2x4x5xf32>

  return %result : vector<2x4x5xf32>
}

// -----

// Already canonical: masked (m, n, k).

//      CHECK-DAG: #[[$MAP_LHS:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//      CHECK-DAG: #[[$MAP_RHS:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//      CHECK-DAG: #[[$MAP_ACC:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func @already_canonical_masked
func.func @already_canonical_masked(
    %A: vector<4x8xf32>,
    %B: vector<6x8xf32>,
    %C: vector<4x6xf32>,
    %mask: vector<4x6x8xi1>) -> vector<4x6xf32> {

  // CHECK-NOT: vector.transpose
  // CHECK: vector.contract
  // CHECK-SAME: indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_ACC]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]

  %result = vector.mask %mask {
    vector.contract {
        indexing_maps = [
            affine_map<(m, n, k) -> (m, k)>,
            affine_map<(m, n, k) -> (n, k)>,
            affine_map<(m, n, k) -> (m, n)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
    } %A, %B, %C : vector<4x8xf32>, vector<6x8xf32> into vector<4x6xf32>
  } : vector<4x6x8xi1> -> vector<4x6xf32>

  return %result : vector<4x6xf32>
}

// -----

// Pure reduction contract: no parallel dims at all. Pattern should not fire.

// CHECK-LABEL: func @pure_reduction_no_parallel_dims
func.func @pure_reduction_no_parallel_dims(
    %A: vector<8xf32>,
    %B: vector<8xf32>,
    %C: f32) -> f32 {

  // CHECK-NOT: vector.transpose
  // CHECK: vector.contract
  // CHECK-SAME: iterator_types = ["reduction"]

  %result = vector.contract {
      indexing_maps = [
          affine_map<(k) -> (k)>,
          affine_map<(k) -> (k)>,
          affine_map<(k) -> ()>
      ],
      iterator_types = ["reduction"]
  } %A, %B, %C : vector<8xf32>, vector<8xf32> into f32

  return %result : f32
}

// -----

// Multiple reduction dims (k1, k2): only parallel dims are reordered;
// reduction dims stay in their original relative order.

//      CHECK-DAG: #[[$MAP_LHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//      CHECK-DAG: #[[$MAP_RHS:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
//      CHECK-DAG: #[[$MAP_ACC:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
// CHECK-LABEL: func @multiple_reduction_dims
// CHECK-SAME: %[[A:.+]]: vector<4x3x8xf32>,
// CHECK-SAME: %[[B:.+]]: vector<3x6x8xf32>,
// CHECK-SAME: %[[C:.+]]: vector<6x4xf32>
func.func @multiple_reduction_dims(
    %A: vector<4x3x8xf32>,
    %B: vector<3x6x8xf32>,
    %C: vector<6x4xf32>) -> vector<6x4xf32> {

  // CHECK-DAG: %[[BT:.+]] = vector.transpose %[[B]], [1, 0, 2] : vector<3x6x8xf32> to vector<6x3x8xf32>
  // CHECK-DAG: %[[CT:.+]] = vector.transpose %[[C]], [1, 0] : vector<6x4xf32> to vector<4x6xf32>
  // CHECK: %[[RES:.+]] = vector.contract
  // CHECK-SAME: indexing_maps = [#[[$MAP_LHS]], #[[$MAP_RHS]], #[[$MAP_ACC]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  // CHECK-SAME: %[[A]], %[[BT]], %[[CT]]
  // CHECK-SAME: : vector<4x3x8xf32>, vector<6x3x8xf32> into vector<4x6xf32>
  // CHECK: vector.transpose %[[RES]], [1, 0] : vector<4x6xf32> to vector<6x4xf32>

  %result = vector.contract {
      indexing_maps = [
          affine_map<(n, m, k1, k2) -> (m, k1, k2)>,
          affine_map<(n, m, k1, k2) -> (k1, n, k2)>,
          affine_map<(n, m, k1, k2) -> (n, m)>
      ],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } %A, %B, %C : vector<4x3x8xf32>, vector<3x6x8xf32> into vector<6x4xf32>

  return %result : vector<6x4xf32>
}

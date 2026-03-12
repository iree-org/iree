// RUN: iree-opt --pass-pipeline="builtin.module(iree-codegen-test-contract-unrolling-patterns)" --split-input-file %s | FileCheck %s

//===----------------------------------------------------------------------===//
// UnrollContractAlongBatchDim
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @unroll_contract_batch_matmul
// CHECK-SAME: %[[A:.+]]: vector<2x4x3xf32>,
// CHECK-SAME: %[[B:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[C:.+]]: vector<2x4x5xf32>
func.func @unroll_contract_batch_matmul(
    %A: vector<2x4x3xf32>,
    %B: vector<2x3x5xf32>,
    %C: vector<2x4x5xf32>) -> vector<2x4x5xf32> {

  // After batch unrolling, B is extracted per batch.
  // After LHS free dim unrolling, B slices are reused across m iterations.
  // CHECK-DAG: %[[B0:.+]] = vector.extract %[[B]][0] : vector<3x5xf32> from vector<2x3x5xf32>
  // CHECK-DAG: %[[B1:.+]] = vector.extract %[[B]][1] : vector<3x5xf32> from vector<2x3x5xf32>

  // Extracts for batch 0, m=0,1,2,3
  // CHECK-DAG: %[[A00:.+]] = vector.extract %[[A]][0, 0] : vector<3xf32> from vector<2x4x3xf32>
  // CHECK-DAG: %[[C00:.+]] = vector.extract %[[C]][0, 0] : vector<5xf32> from vector<2x4x5xf32>
  // CHECK-DAG: %[[A01:.+]] = vector.extract %[[A]][0, 1] : vector<3xf32> from vector<2x4x3xf32>
  // CHECK-DAG: %[[C01:.+]] = vector.extract %[[C]][0, 1] : vector<5xf32> from vector<2x4x5xf32>
  // CHECK-DAG: %[[A02:.+]] = vector.extract %[[A]][0, 2] : vector<3xf32> from vector<2x4x3xf32>
  // CHECK-DAG: %[[C02:.+]] = vector.extract %[[C]][0, 2] : vector<5xf32> from vector<2x4x5xf32>
  // CHECK-DAG: %[[A03:.+]] = vector.extract %[[A]][0, 3] : vector<3xf32> from vector<2x4x3xf32>
  // CHECK-DAG: %[[C03:.+]] = vector.extract %[[C]][0, 3] : vector<5xf32> from vector<2x4x5xf32>

  // Final contracts: vector<3xf32>, vector<3x5xf32> -> vector<5xf32>
  // CHECK: vector.contract
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]
  // CHECK-SAME: %[[A00]], %[[B0]], %[[C00]]
  // CHECK-SAME: : vector<3xf32>, vector<3x5xf32> into vector<5xf32>

  // CHECK: vector.contract
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]
  // CHECK-SAME: %[[A01]], %[[B0]], %[[C01]]

  // CHECK: vector.contract
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]
  // CHECK-SAME: %[[A02]], %[[B0]], %[[C02]]

  // CHECK: vector.contract
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]
  // CHECK-SAME: %[[A03]], %[[B0]], %[[C03]]

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

// CHECK-LABEL: func @unroll_contract_batch_masked
// CHECK-SAME: %[[A:.+]]: vector<2x4x3xf32>,
// CHECK-SAME: %[[B:.+]]: vector<2x3x5xf32>,
// CHECK-SAME: %[[C:.+]]: vector<2x4x5xf32>,
// CHECK-SAME: %[[MASK:.+]]: vector<2x4x5x3xi1>
func.func @unroll_contract_batch_masked(
    %A: vector<2x4x3xf32>,
    %B: vector<2x3x5xf32>,
    %C: vector<2x4x5xf32>,
    %mask: vector<2x4x5x3xi1>) -> vector<2x4x5xf32> {

  // CHECK-DAG: %[[B0:.+]] = vector.extract %[[B]][0] : vector<3x5xf32> from vector<2x3x5xf32>
  // CHECK-DAG: %[[A00:.+]] = vector.extract %[[A]][0, 0] : vector<3xf32> from vector<2x4x3xf32>
  // CHECK-DAG: %[[C00:.+]] = vector.extract %[[C]][0, 0] : vector<5xf32> from vector<2x4x5xf32>
  // CHECK-DAG: %[[MASK00:.+]] = vector.extract %[[MASK]][0, 0] : vector<5x3xi1> from vector<2x4x5x3xi1>

  // CHECK: vector.mask %[[MASK00]] {
  // CHECK:   vector.contract {{.*}} %[[A00]], %[[B0]], %[[C00]]
  // CHECK:   : vector<3xf32>, vector<3x5xf32> into vector<5xf32>
  // CHECK: } : vector<5x3xi1> -> vector<5xf32>

  %result = vector.mask %mask {
    vector.contract {
        indexing_maps = [
            affine_map<(b, m, n, k) -> (b, m, k)>,
            affine_map<(b, m, n, k) -> (b, k, n)>,
            affine_map<(b, m, n, k) -> (b, m, n)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } %A, %B, %C : vector<2x4x3xf32>, vector<2x3x5xf32> into vector<2x4x5xf32>
  } : vector<2x4x5x3xi1> -> vector<2x4x5xf32>

  return %result : vector<2x4x5xf32>
}

// -----

//===----------------------------------------------------------------------===//
// UnrollContractAlongLhsFreeDim
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @unroll_contract_lhs_free_matmul
// CHECK-SAME: %[[A:.+]]: vector<4x8xf32>,
// CHECK-SAME: %[[B:.+]]: vector<8x6xf32>,
// CHECK-SAME: %[[C:.+]]: vector<4x6xf32>
func.func @unroll_contract_lhs_free_matmul(
    %A: vector<4x8xf32>,
    %B: vector<8x6xf32>,
    %C: vector<4x6xf32>) -> vector<4x6xf32> {

  // CHECK-DAG: %[[A0:.+]] = vector.extract %[[A]][0] : vector<8xf32> from vector<4x8xf32>
  // CHECK-DAG: %[[A1:.+]] = vector.extract %[[A]][1] : vector<8xf32> from vector<4x8xf32>
  // CHECK-DAG: %[[A2:.+]] = vector.extract %[[A]][2] : vector<8xf32> from vector<4x8xf32>
  // CHECK-DAG: %[[A3:.+]] = vector.extract %[[A]][3] : vector<8xf32> from vector<4x8xf32>

  // CHECK-DAG: %[[C0:.+]] = vector.extract %[[C]][0] : vector<6xf32> from vector<4x6xf32>
  // CHECK-DAG: %[[C1:.+]] = vector.extract %[[C]][1] : vector<6xf32> from vector<4x6xf32>
  // CHECK-DAG: %[[C2:.+]] = vector.extract %[[C]][2] : vector<6xf32> from vector<4x6xf32>
  // CHECK-DAG: %[[C3:.+]] = vector.extract %[[C]][3] : vector<6xf32> from vector<4x6xf32>

  // CHECK-NOT: vector.extract %[[B]]

  // CHECK: %[[R0:.+]] = vector.contract
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]
  // CHECK-SAME: %[[A0]], %[[B]], %[[C0]]
  // CHECK-SAME: : vector<8xf32>, vector<8x6xf32> into vector<6xf32>

  // CHECK: %[[R1:.+]] = vector.contract
  // CHECK-SAME: %[[A1]], %[[B]], %[[C1]]
  // CHECK-SAME: : vector<8xf32>, vector<8x6xf32> into vector<6xf32>

  // CHECK: %[[R2:.+]] = vector.contract
  // CHECK-SAME: %[[A2]], %[[B]], %[[C2]]
  // CHECK-SAME: : vector<8xf32>, vector<8x6xf32> into vector<6xf32>

  // CHECK: %[[R3:.+]] = vector.contract
  // CHECK-SAME: %[[A3]], %[[B]], %[[C3]]
  // CHECK-SAME: : vector<8xf32>, vector<8x6xf32> into vector<6xf32>

  // CHECK: vector.insert %[[R0]], {{.*}}[0]
  // CHECK: vector.insert %[[R1]], {{.*}}[1]
  // CHECK: vector.insert %[[R2]], {{.*}}[2]
  // CHECK: vector.insert %[[R3]], {{.*}}[3]

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

// CHECK-LABEL: func @unroll_contract_lhs_free_matvec
// CHECK-SAME: %[[A:.+]]: vector<3x8xf32>,
// CHECK-SAME: %[[B:.+]]: vector<8xf32>,
// CHECK-SAME: %[[C:.+]]: vector<3xf32>
func.func @unroll_contract_lhs_free_matvec(
    %A: vector<3x8xf32>,
    %B: vector<8xf32>,
    %C: vector<3xf32>) -> vector<3xf32> {

  // CHECK-DAG: %[[A0:.+]] = vector.extract %[[A]][0] : vector<8xf32> from vector<3x8xf32>
  // CHECK-DAG: %[[A1:.+]] = vector.extract %[[A]][1] : vector<8xf32> from vector<3x8xf32>
  // CHECK-DAG: %[[A2:.+]] = vector.extract %[[A]][2] : vector<8xf32> from vector<3x8xf32>

  // CHECK-DAG: %[[C0:.+]] = vector.extract %[[C]][0] : f32 from vector<3xf32>
  // CHECK-DAG: %[[C1:.+]] = vector.extract %[[C]][1] : f32 from vector<3xf32>
  // CHECK-DAG: %[[C2:.+]] = vector.extract %[[C]][2] : f32 from vector<3xf32>

  // CHECK-NOT: vector.extract %[[B]]

  // Inner contracts are pure-reduction (scalar result), kept as vector.contract
  // CHECK: %[[R0:.+]] = vector.contract
  // CHECK-SAME: iterator_types = ["reduction"]
  // CHECK-SAME: %[[A0]], %[[B]], %[[C0]]
  // CHECK-SAME: : vector<8xf32>, vector<8xf32> into f32

  // CHECK: %[[R1:.+]] = vector.contract
  // CHECK-SAME: iterator_types = ["reduction"]
  // CHECK-SAME: %[[A1]], %[[B]], %[[C1]]
  // CHECK-SAME: : vector<8xf32>, vector<8xf32> into f32

  // CHECK: %[[R2:.+]] = vector.contract
  // CHECK-SAME: iterator_types = ["reduction"]
  // CHECK-SAME: %[[A2]], %[[B]], %[[C2]]
  // CHECK-SAME: : vector<8xf32>, vector<8xf32> into f32

  // CHECK: vector.insert %[[R0]], {{.*}}[0]
  // CHECK: vector.insert %[[R1]], {{.*}}[1]
  // CHECK: vector.insert %[[R2]], {{.*}}[2]

  %result = vector.contract {
      indexing_maps = [
          affine_map<(m, k) -> (m, k)>,
          affine_map<(m, k) -> (k)>,
          affine_map<(m, k) -> (m)>
      ],
      iterator_types = ["parallel", "reduction"]
  } %A, %B, %C : vector<3x8xf32>, vector<8xf32> into vector<3xf32>

  return %result : vector<3xf32>
}

// -----

// CHECK-LABEL: func @unroll_contract_lhs_free_masked
// CHECK-SAME: %[[A:.+]]: vector<3x8xf32>,
// CHECK-SAME: %[[B:.+]]: vector<8x5xf32>,
// CHECK-SAME: %[[C:.+]]: vector<3x5xf32>,
// CHECK-SAME: %[[MASK:.+]]: vector<3x5x8xi1>
func.func @unroll_contract_lhs_free_masked(
    %A: vector<3x8xf32>,
    %B: vector<8x5xf32>,
    %C: vector<3x5xf32>,
    %mask: vector<3x5x8xi1>) -> vector<3x5xf32> {

  // CHECK-DAG: %[[A0:.+]] = vector.extract %[[A]][0]
  // CHECK-DAG: %[[A1:.+]] = vector.extract %[[A]][1]
  // CHECK-DAG: %[[A2:.+]] = vector.extract %[[A]][2]

  // CHECK-DAG: %[[C0:.+]] = vector.extract %[[C]][0]
  // CHECK-DAG: %[[C1:.+]] = vector.extract %[[C]][1]
  // CHECK-DAG: %[[C2:.+]] = vector.extract %[[C]][2]

  // CHECK-DAG: %[[MASK0:.+]] = vector.extract %[[MASK]][0]
  // CHECK-DAG: %[[MASK1:.+]] = vector.extract %[[MASK]][1]
  // CHECK-DAG: %[[MASK2:.+]] = vector.extract %[[MASK]][2]

  // CHECK-NOT: vector.extract %[[B]]

  // CHECK: vector.mask %[[MASK0]] {
  // CHECK:   vector.contract {{.*}} %[[A0]], %[[B]], %[[C0]]
  // CHECK: }

  // CHECK: vector.mask %[[MASK1]] {
  // CHECK:   vector.contract {{.*}} %[[A1]], %[[B]], %[[C1]]
  // CHECK: }

  // CHECK: vector.mask %[[MASK2]] {
  // CHECK:   vector.contract {{.*}} %[[A2]], %[[B]], %[[C2]]
  // CHECK: }

  %result = vector.mask %mask {
    vector.contract {
        indexing_maps = [
            affine_map<(m, n, k) -> (m, k)>,
            affine_map<(m, n, k) -> (k, n)>,
            affine_map<(m, n, k) -> (m, n)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
    } %A, %B, %C : vector<3x8xf32>, vector<8x5xf32> into vector<3x5xf32>
  } : vector<3x5x8xi1> -> vector<3x5xf32>

  return %result : vector<3x5xf32>
}

// -----

// Negative test: free LHS dim not at outermost position in LHS
// CHECK-LABEL: func @unroll_contract_lhs_free_not_outermost
func.func @unroll_contract_lhs_free_not_outermost(
    %A: vector<8x4xf32>,
    %B: vector<8x3xf32>,
    %C: vector<4x3xf32>) -> vector<4x3xf32> {

  // CHECK: vector.contract
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK-NOT: vector.extract
  // CHECK: return

  %result = vector.contract {
      indexing_maps = [
          affine_map<(m, n, k) -> (k, m)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
  } %A, %B, %C : vector<8x4xf32>, vector<8x3xf32> into vector<4x3xf32>

  return %result : vector<4x3xf32>
}

// -----

//===----------------------------------------------------------------------===//
// UnrollContractAlongRhsFreeDim
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @unroll_contract_rhs_free_basic
// CHECK-SAME: %[[A:.+]]: vector<4x8xf32>,
// CHECK-SAME: %[[B:.+]]: vector<6x8xf32>,
// CHECK-SAME: %[[C:.+]]: vector<6x4xf32>
func.func @unroll_contract_rhs_free_basic(
    %A: vector<4x8xf32>,
    %B: vector<6x8xf32>,
    %C: vector<6x4xf32>) -> vector<6x4xf32> {

  // 1. UnrollContractAlongRhsFreeDim unrolls along n (size 6)
  // 2. UnrollContractAlongLhsFreeDim unrolls along m (size 4) within each

  // CHECK-DAG: %[[B0:.+]] = vector.extract %[[B]][0] : vector<8xf32> from vector<6x8xf32>
  // CHECK-DAG: %[[B1:.+]] = vector.extract %[[B]][1] : vector<8xf32> from vector<6x8xf32>
  // CHECK-DAG: %[[B2:.+]] = vector.extract %[[B]][2] : vector<8xf32> from vector<6x8xf32>
  // CHECK-DAG: %[[B3:.+]] = vector.extract %[[B]][3] : vector<8xf32> from vector<6x8xf32>
  // CHECK-DAG: %[[B4:.+]] = vector.extract %[[B]][4] : vector<8xf32> from vector<6x8xf32>
  // CHECK-DAG: %[[B5:.+]] = vector.extract %[[B]][5] : vector<8xf32> from vector<6x8xf32>

  // CHECK-DAG: %[[A0:.+]] = vector.extract %[[A]][0] : vector<8xf32> from vector<4x8xf32>
  // CHECK-DAG: %[[A1:.+]] = vector.extract %[[A]][1] : vector<8xf32> from vector<4x8xf32>
  // CHECK-DAG: %[[A2:.+]] = vector.extract %[[A]][2] : vector<8xf32> from vector<4x8xf32>
  // CHECK-DAG: %[[A3:.+]] = vector.extract %[[A]][3] : vector<8xf32> from vector<4x8xf32>

  // CHECK-DAG: vector.extract %[[C]][0, 0] : f32 from vector<6x4xf32>
  // CHECK-DAG: vector.extract %[[C]][0, 1] : f32 from vector<6x4xf32>

  // Innermost contracts are pure-reduction, kept as vector.contract
  // CHECK: vector.contract
  // CHECK-SAME: iterator_types = ["reduction"]
  // CHECK-SAME: : vector<8xf32>, vector<8xf32> into f32

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

// CHECK-LABEL: func @unroll_contract_rhs_free_vecmat
// CHECK-SAME: %[[A:.+]]: vector<8xf32>,
// CHECK-SAME: %[[B:.+]]: vector<3x8xf32>,
// CHECK-SAME: %[[C:.+]]: vector<3xf32>
func.func @unroll_contract_rhs_free_vecmat(
    %A: vector<8xf32>,
    %B: vector<3x8xf32>,
    %C: vector<3xf32>) -> vector<3xf32> {

  // CHECK-DAG: %[[B0:.+]] = vector.extract %[[B]][0] : vector<8xf32> from vector<3x8xf32>
  // CHECK-DAG: %[[B1:.+]] = vector.extract %[[B]][1] : vector<8xf32> from vector<3x8xf32>
  // CHECK-DAG: %[[B2:.+]] = vector.extract %[[B]][2] : vector<8xf32> from vector<3x8xf32>

  // CHECK-DAG: %[[C0:.+]] = vector.extract %[[C]][0] : f32 from vector<3xf32>
  // CHECK-DAG: %[[C1:.+]] = vector.extract %[[C]][1] : f32 from vector<3xf32>
  // CHECK-DAG: %[[C2:.+]] = vector.extract %[[C]][2] : f32 from vector<3xf32>

  // Inner contracts are pure-reduction, kept as vector.contract
  // CHECK: vector.contract
  // CHECK-SAME: iterator_types = ["reduction"]
  // CHECK-SAME: %[[A]], %[[B0]], %[[C0]]
  // CHECK-SAME: : vector<8xf32>, vector<8xf32> into f32

  // CHECK: vector.contract
  // CHECK-SAME: iterator_types = ["reduction"]
  // CHECK-SAME: %[[A]], %[[B1]], %[[C1]]
  // CHECK-SAME: : vector<8xf32>, vector<8xf32> into f32

  // CHECK: vector.contract
  // CHECK-SAME: iterator_types = ["reduction"]
  // CHECK-SAME: %[[A]], %[[B2]], %[[C2]]
  // CHECK-SAME: : vector<8xf32>, vector<8xf32> into f32

  %result = vector.contract {
      indexing_maps = [
          affine_map<(n, k) -> (k)>,
          affine_map<(n, k) -> (n, k)>,
          affine_map<(n, k) -> (n)>
      ],
      iterator_types = ["parallel", "reduction"]
  } %A, %B, %C : vector<8xf32>, vector<3x8xf32> into vector<3xf32>

  return %result : vector<3xf32>
}

// -----

// CHECK-LABEL: func @unroll_contract_rhs_free_masked
// CHECK-SAME: %[[A:.+]]: vector<4x8xf32>,
// CHECK-SAME: %[[B:.+]]: vector<3x8xf32>,
// CHECK-SAME: %[[C:.+]]: vector<3x4xf32>,
// CHECK-SAME: %[[MASK:.+]]: vector<3x4x8xi1>
func.func @unroll_contract_rhs_free_masked(
    %A: vector<4x8xf32>,
    %B: vector<3x8xf32>,
    %C: vector<3x4xf32>,
    %mask: vector<3x4x8xi1>) -> vector<3x4xf32> {

  // CHECK-DAG: %[[B0:.+]] = vector.extract %[[B]][0]
  // CHECK-DAG: %[[B1:.+]] = vector.extract %[[B]][1]
  // CHECK-DAG: %[[B2:.+]] = vector.extract %[[B]][2]

  // CHECK-DAG: %[[A0:.+]] = vector.extract %[[A]][0]
  // CHECK-DAG: %[[A1:.+]] = vector.extract %[[A]][1]

  // CHECK-DAG: %[[MASK00:.+]] = vector.extract %[[MASK]][0, 0]

  // Innermost contracts are pure-reduction with masks, kept as vector.contract
  // CHECK: vector.mask %[[MASK00]] {
  // CHECK:   vector.contract {{.*}} iterator_types = ["reduction"]
  // CHECK:   : vector<8xf32>, vector<8xf32> into f32
  // CHECK: }

  %result = vector.mask %mask {
    vector.contract {
        indexing_maps = [
            affine_map<(n, m, k) -> (m, k)>,
            affine_map<(n, m, k) -> (n, k)>,
            affine_map<(n, m, k) -> (n, m)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
    } %A, %B, %C : vector<4x8xf32>, vector<3x8xf32> into vector<3x4xf32>
  } : vector<3x4x8xi1> -> vector<3x4xf32>

  return %result : vector<3x4xf32>
}


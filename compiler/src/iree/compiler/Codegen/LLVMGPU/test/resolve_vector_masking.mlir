// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-llvmgpu-resolve-vector-masking))" %s | FileCheck %s

// CHECK-LABEL: func.func @unwrap_masked_matmul_add(
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<8x16xf32>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<16x8xf32>
// CHECK-SAME: %[[ACC:[a-zA-Z0-9]+]]: vector<8x8xf32>
// CHECK-SAME: %[[M:[a-zA-Z0-9]+]]: index
// CHECK-SAME: %[[N:[a-zA-Z0-9]+]]: index
// CHECK-SAME: %[[K:[a-zA-Z0-9]+]]: index
func.func @unwrap_masked_matmul_add(%lhs: vector<8x16xf32>, %rhs: vector<16x8xf32>, %acc: vector<8x8xf32>, %m: index, %n: index, %k: index) -> vector<8x8xf32> {
  // CHECK-DAG: %[[IDENTITY_LHS:cst.*]] = arith.constant dense<0.000000e+00>
  // CHECK-DAG: %[[IDENTITY_RHS:cst.*]] = arith.constant dense<0.000000e+00>
  // CHECK: %[[LHS_MASK:.+]] = vector.create_mask %[[M]], %[[K]]
  // CHECK: %[[RHS_MASK:.+]] = vector.create_mask %[[K]], %[[N]]
  // CHECK: %[[LHS_MASKED:.+]] = arith.select %[[LHS_MASK]], %[[LHS]], %[[IDENTITY_LHS]]
  // CHECK: %[[RHS_MASKED:.+]] = arith.select %[[RHS_MASK]], %[[RHS]], %[[IDENTITY_RHS]]
  // CHECK: vector.contract {indexing_maps = {{.+}}, iterator_types = {{.+}}, kind = #vector.kind<add>} %[[LHS_MASKED]], %[[RHS_MASKED]], %[[ACC]]
  // CHECK-NOT: vector.mask

  %mask = vector.create_mask %m, %n, %k : vector<8x8x16xi1>
  %result = vector.mask %mask {
    vector.contract {
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (k, n)>,
                       affine_map<(m, n, k) -> (m, n)>],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %lhs, %rhs, %acc : vector<8x16xf32>, vector<16x8xf32> into vector<8x8xf32>
  } : vector<8x8x16xi1> -> vector<8x8xf32>
  return %result : vector<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @unwrap_masked_matmul_transposed_rhs(
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<8x16xf32>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<8x16xf32>
// CHECK-SAME: %[[ACC:[a-zA-Z0-9]+]]: vector<8x8xf32>
// CHECK-SAME: %[[M:[a-zA-Z0-9]+]]: index
// CHECK-SAME: %[[N:[a-zA-Z0-9]+]]: index
// CHECK-SAME: %[[K:[a-zA-Z0-9]+]]: index
func.func @unwrap_masked_matmul_transposed_rhs(%lhs: vector<8x16xf32>, %rhs: vector<8x16xf32>, %acc: vector<8x8xf32>, %m: index, %n: index, %k: index) -> vector<8x8xf32> {
  // CHECK-DAG: %[[IDENTITY_1:cst.*]] = arith.constant dense<0.000000e+00>
  // CHECK-DAG: %[[IDENTITY_2:cst.*]] = arith.constant dense<0.000000e+00>
  // CHECK: %[[LHS_MASK:.+]] = vector.create_mask %[[M]], %[[K]]
  // CHECK: %[[RHS_MASK:.+]] = vector.create_mask %[[N]], %[[K]]
  // CHECK: %[[LHS_MASKED:.+]] = arith.select %[[LHS_MASK]], %[[LHS]], %[[IDENTITY_1]]
  // CHECK: %[[RHS_MASKED:.+]] = arith.select %[[RHS_MASK]], %[[RHS]], %[[IDENTITY_2]]
  // CHECK: vector.contract {indexing_maps = {{.+}}, iterator_types = {{.+}}, kind = #vector.kind<add>} %[[LHS_MASKED]], %[[RHS_MASKED]], %[[ACC]]
  // CHECK-NOT: vector.mask

  %mask = vector.create_mask %m, %n, %k : vector<8x8x16xi1>
  %result = vector.mask %mask {
    vector.contract {
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (m, n)>],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %lhs, %rhs, %acc : vector<8x16xf32>, vector<8x16xf32> into vector<8x8xf32>
  } : vector<8x8x16xi1> -> vector<8x8xf32>
  return %result : vector<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @unwrap_masked_matmul_mul
func.func @unwrap_masked_matmul_mul(%lhs: vector<8x16xf32>, %rhs: vector<16x8xf32>, %acc: vector<8x8xf32>, %m: index, %n: index, %k: index) -> vector<8x8xf32> {
  // CHECK-DAG: arith.constant dense<1.000000e+00> : vector<8x16xf32>
  // CHECK-DAG: arith.constant dense<1.000000e+00> : vector<16x8xf32>
  // CHECK: vector.contract
  // CHECK-SAME: kind = #vector.kind<mul>
  // CHECK-NOT: vector.mask

  %mask = vector.create_mask %m, %n, %k : vector<8x8x16xi1>
  %result = vector.mask %mask {
    vector.contract {
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (k, n)>,
                       affine_map<(m, n, k) -> (m, n)>],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<mul>
    } %lhs, %rhs, %acc : vector<8x16xf32>, vector<16x8xf32> into vector<8x8xf32>
  } : vector<8x8x16xi1> -> vector<8x8xf32>
  return %result : vector<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @preserve_attributes
func.func @preserve_attributes(%lhs: vector<8x16xf32>, %rhs: vector<16x8xf32>, %acc: vector<8x8xf32>, %m: index, %n: index, %k: index) -> vector<8x8xf32> {
  // CHECK: vector.contract
  // CHECK-SAME: iree.test.attr = "preserved"
  // CHECK-NOT: vector.mask

  %mask = vector.create_mask %m, %n, %k : vector<8x8x16xi1>
  %result = vector.mask %mask {
    vector.contract {
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (k, n)>,
                       affine_map<(m, n, k) -> (m, n)>],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>,
      iree.test.attr = "preserved"
    } %lhs, %rhs, %acc : vector<8x16xf32>, vector<16x8xf32> into vector<8x8xf32>
  } : vector<8x8x16xi1> -> vector<8x8xf32>
  return %result : vector<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @attention_like_contract_f16_f32(
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<64x64xf16>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<16x64xf16>
// CHECK-SAME: %[[ACC:[a-zA-Z0-9]+]]: vector<16x64xf32>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: index
func.func @attention_like_contract_f16_f32(
    %lhs: vector<64x64xf16>,
    %rhs: vector<16x64xf16>,
    %acc: vector<16x64xf32>,
    %arg1: index) -> vector<16x64xf32> {
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index

  // CHECK: %[[C16:.+]] = arith.constant 16
  // CHECK: %[[C64:.+]] = arith.constant 64
  // CHECK: %[[BOUND:.+]] = affine.min {{.+}}(%[[ARG1]])
  %bound = affine.min affine_map<(d0) -> (-d0 + 4080, 64)>(%arg1)

  // CHECK-DAG: %[[IDENTITY_LHS:cst.*]] = arith.constant dense<0.000000e+00> : vector<64x64xf16>
  // CHECK-DAG: %[[IDENTITY_RHS:cst.*]] = arith.constant dense<0.000000e+00> : vector<16x64xf16>

  // CHECK: %[[LHS_MASK:.+]] = vector.create_mask %[[BOUND]], %[[C64]]
  // CHECK: %[[RHS_MASK:.+]] = vector.create_mask %[[C16]], %[[C64]]
  // CHECK: %[[LHS_MASKED:.+]] = arith.select %[[LHS_MASK]], %[[LHS]], %[[IDENTITY_LHS]]
  // CHECK: %[[RHS_MASKED:.+]] = arith.select %[[RHS_MASK]], %[[RHS]], %[[IDENTITY_RHS]]
  // CHECK: vector.contract {indexing_maps = {{.+}}, iterator_types = {{.+}}, kind = #vector.kind<add>} %[[LHS_MASKED]], %[[RHS_MASKED]], %[[ACC]]
  // CHECK-NOT: vector.mask

  // Create mask with affine.min result
  %mask = vector.create_mask %c16, %c64, %bound : vector<16x64x64xi1>
  %result = vector.mask %mask {
    vector.contract {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d2)>],
      iterator_types = ["parallel", "reduction", "parallel"],
      kind = #vector.kind<add>
    } %lhs, %rhs, %acc : vector<64x64xf16>, vector<16x64xf16> into vector<16x64xf32>
  } : vector<16x64x64xi1> -> vector<16x64xf32>
  return %result : vector<16x64xf32>
}

// -----

// CHECK-LABEL: func.func @no_unwrap_non_create_mask
// Negative test: mask from block argument should not be unwrapped
func.func @no_unwrap_non_create_mask(
    %lhs: vector<8x16xf32>,
    %rhs: vector<16x8xf32>,
    %acc: vector<8x8xf32>,
    %mask: vector<8x8x16xi1>) -> vector<8x8xf32> {
  // CHECK: vector.mask
  // CHECK-SAME: vector.contract
  %result = vector.mask %mask {
    vector.contract {
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (k, n)>,
                       affine_map<(m, n, k) -> (m, n)>],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %lhs, %rhs, %acc : vector<8x16xf32>, vector<16x8xf32> into vector<8x8xf32>
  } : vector<8x8x16xi1> -> vector<8x8xf32>
  return %result : vector<8x8xf32>
}

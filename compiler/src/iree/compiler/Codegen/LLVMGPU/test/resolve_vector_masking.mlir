// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-llvmgpu-resolve-vector-masking))" %s | FileCheck %s --implicit-check-not="vector.mask"

// CHECK-LABEL: func.func @unwrap_masked_matmul_add(
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<8x16xf32>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<16x8xf32>
// CHECK-SAME: %[[ACC:[a-zA-Z0-9]+]]: vector<8x8xf32>
// CHECK-SAME: %[[M:[a-zA-Z0-9]+]]: index
// CHECK-SAME: %[[N:[a-zA-Z0-9]+]]: index
// CHECK-SAME: %[[K:[a-zA-Z0-9]+]]: index
func.func @unwrap_masked_matmul_add(%lhs: vector<8x16xf32>, %rhs: vector<16x8xf32>, %acc: vector<8x8xf32>, %m: index, %n: index, %k: index) -> vector<8x8xf32> {
  // LHS mask: decomposed from create_mask(%m, %k) on <8x16xi1>
  // CHECK-DAG: %[[IDENTITY_LHS:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
  // CHECK-DAG: %[[IDENTITY_RHS:.*]] = arith.constant dense<0.000000e+00> : vector<16x8xf32>
  // CHECK: %[[STEP_M:.+]] = vector.step : vector<8xindex>
  // CHECK: %[[BCAST_M:.+]] = vector.broadcast %[[M]] : index to vector<8xindex>
  // CHECK: %[[CMP_M:.+]] = arith.cmpi slt, %[[STEP_M]], %[[BCAST_M]] : vector<8xindex>
  // CHECK: %[[BCAST_CMP_M:.+]] = vector.broadcast %[[CMP_M]] : vector<8xi1> to vector<16x8xi1>
  // CHECK: %[[TRANS_M:.+]] = vector.transpose %[[BCAST_CMP_M]], [1, 0] : vector<16x8xi1> to vector<8x16xi1>
  // CHECK: %[[STEP_K:.+]] = vector.step : vector<16xindex>
  // CHECK: %[[BCAST_K:.+]] = vector.broadcast %[[K]] : index to vector<16xindex>
  // CHECK: %[[CMP_K:.+]] = arith.cmpi slt, %[[STEP_K]], %[[BCAST_K]] : vector<16xindex>
  // CHECK: %[[BCAST_CMP_K:.+]] = vector.broadcast %[[CMP_K]] : vector<16xi1> to vector<8x16xi1>
  // CHECK: %[[LHS_MASK:.+]] = arith.andi %[[TRANS_M]], %[[BCAST_CMP_K]] : vector<8x16xi1>
  // RHS mask: decomposed from create_mask(%k, %n) on <16x8xi1>
  // CHECK: %[[STEP_K2:.+]] = vector.step : vector<16xindex>
  // CHECK: %[[BCAST_K2:.+]] = vector.broadcast %[[K]] : index to vector<16xindex>
  // CHECK: %[[CMP_K2:.+]] = arith.cmpi slt, %[[STEP_K2]], %[[BCAST_K2]] : vector<16xindex>
  // CHECK: %[[BCAST_CMP_K2:.+]] = vector.broadcast %[[CMP_K2]] : vector<16xi1> to vector<8x16xi1>
  // CHECK: %[[TRANS_K2:.+]] = vector.transpose %[[BCAST_CMP_K2]], [1, 0] : vector<8x16xi1> to vector<16x8xi1>
  // CHECK: %[[STEP_N:.+]] = vector.step : vector<8xindex>
  // CHECK: %[[BCAST_N:.+]] = vector.broadcast %[[N]] : index to vector<8xindex>
  // CHECK: %[[CMP_N:.+]] = arith.cmpi slt, %[[STEP_N]], %[[BCAST_N]] : vector<8xindex>
  // CHECK: %[[BCAST_CMP_N:.+]] = vector.broadcast %[[CMP_N]] : vector<8xi1> to vector<16x8xi1>
  // CHECK: %[[RHS_MASK:.+]] = arith.andi %[[TRANS_K2]], %[[BCAST_CMP_N]] : vector<16x8xi1>
  // CHECK: %[[LHS_MASKED:.+]] = arith.select %[[LHS_MASK]], %[[LHS]], %[[IDENTITY_LHS]]
  // CHECK: %[[RHS_MASKED:.+]] = arith.select %[[RHS_MASK]], %[[RHS]], %[[IDENTITY_RHS]]
  // CHECK: vector.contract {indexing_maps = {{.+}}, iterator_types = {{.+}}, kind = #vector.kind<add>} %[[LHS_MASKED]], %[[RHS_MASKED]], %[[ACC]]

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
  // Both LHS and RHS masks are <8x16xi1>. LHS: create_mask(%m, %k), RHS: create_mask(%n, %k).
  // CHECK-DAG: %[[IDENTITY:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
  // CHECK: arith.cmpi slt
  // CHECK: arith.cmpi slt
  // CHECK: %[[LHS_MASK:.+]] = arith.andi {{.+}} : vector<8x16xi1>
  // CHECK: arith.cmpi slt
  // CHECK: arith.cmpi slt
  // CHECK: %[[RHS_MASK:.+]] = arith.andi {{.+}} : vector<8x16xi1>
  // CHECK: %[[LHS_MASKED:.+]] = arith.select %[[LHS_MASK]], %[[LHS]], %[[IDENTITY]]
  // CHECK: %[[RHS_MASKED:.+]] = arith.select %[[RHS_MASK]], %[[RHS]], %[[IDENTITY]]
  // CHECK: vector.contract {{.+}} %[[LHS_MASKED]], %[[RHS_MASKED]], %[[ACC]]

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
  // CHECK: vector.contract
  // CHECK-SAME: kind = #vector.kind<mul>

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

// Test: attention-like contract where the dynamic dim only appears in the LHS
// indexing map. The RHS mask is all-true and should not get a select.
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

  // CHECK: %[[BOUND:.+]] = affine.min {{.+}}(%[[ARG1]])
  %bound = affine.min affine_map<(d0) -> (-d0 + 4080, 64)>(%arg1)

  // CHECK: %[[STEP:.+]] = vector.step : vector<64xindex>
  // CHECK: %[[BCAST_BOUND:.+]] = vector.broadcast %[[BOUND]] : index to vector<64xindex>
  // CHECK: %[[CMP:.+]] = arith.cmpi slt, %[[STEP]], %[[BCAST_BOUND]] : vector<64xindex>
  // CHECK: %[[BCAST_CMP:.+]] = vector.broadcast %[[CMP]] : vector<64xi1> to vector<64x64xi1>
  // CHECK: %[[LHS_MASK:.+]] = vector.transpose %[[BCAST_CMP]], [1, 0] : vector<64x64xi1> to vector<64x64xi1>
  // CHECK: %[[LHS_MASKED:.+]] = arith.select %[[LHS_MASK]], %[[LHS]], %{{.*}}
  // CHECK: vector.contract {{.+}} %[[LHS_MASKED]], %[[RHS]], %[[ACC]]

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

// CHECK-LABEL: func.func @unwrap_non_create_mask(
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<8x16xf32>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<16x8xf32>
// CHECK-SAME: %[[ACC:[a-zA-Z0-9]+]]: vector<8x8xf32>
// CHECK-SAME: %[[MASK:[a-zA-Z0-9]+]]: vector<8x8x16xi1>
func.func @unwrap_non_create_mask(
    %lhs: vector<8x16xf32>,
    %rhs: vector<16x8xf32>,
    %acc: vector<8x8xf32>,
    %mask: vector<8x8x16xi1>) -> vector<8x8xf32> {
  // CHECK-DAG: %[[IDENTITY_LHS:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
  // CHECK-DAG: %[[IDENTITY_RHS:.*]] = arith.constant dense<0.000000e+00> : vector<16x8xf32>
  // CHECK: %[[LHS_T:.+]] = vector.transpose %[[MASK]], [1, 0, 2] : vector<8x8x16xi1> to vector<8x8x16xi1>
  // CHECK: %[[LHS_MASK:.+]] = vector.extract %[[LHS_T]][0] : vector<8x16xi1> from vector<8x8x16xi1>
  // CHECK: %[[RHS_T:.+]] = vector.transpose %[[MASK]], [0, 2, 1] : vector<8x8x16xi1> to vector<8x16x8xi1>
  // CHECK: %[[RHS_MASK:.+]] = vector.extract %[[RHS_T]][0] : vector<16x8xi1> from vector<8x16x8xi1>
  // CHECK: %[[LHS_MASKED:.+]] = arith.select %[[LHS_MASK]], %[[LHS]], %[[IDENTITY_LHS]]
  // CHECK: %[[RHS_MASKED:.+]] = arith.select %[[RHS_MASK]], %[[RHS]], %[[IDENTITY_RHS]]
  // CHECK: vector.contract {{.*}} %[[LHS_MASKED]], %[[RHS_MASKED]], %[[ACC]]

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

// CHECK-LABEL: func.func @unwrap_non_create_mask_transposed_rhs(
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<8x16xf32>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<8x16xf32>
// CHECK-SAME: %[[ACC:[a-zA-Z0-9]+]]: vector<8x8xf32>
// CHECK-SAME: %[[MASK:[a-zA-Z0-9]+]]: vector<8x8x16xi1>
func.func @unwrap_non_create_mask_transposed_rhs(
    %lhs: vector<8x16xf32>,
    %rhs: vector<8x16xf32>,
    %acc: vector<8x8xf32>,
    %mask: vector<8x8x16xi1>) -> vector<8x8xf32> {
  // CHECK-DAG: %[[IDENTITY:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
  // CHECK: %[[LHS_T:.+]] = vector.transpose %[[MASK]], [1, 0, 2] : vector<8x8x16xi1> to vector<8x8x16xi1>
  // CHECK: %[[LHS_MASK:.+]] = vector.extract %[[LHS_T]][0] : vector<8x16xi1> from vector<8x8x16xi1>
  // CHECK: %[[RHS_MASK:.+]] = vector.extract %[[MASK]][0] : vector<8x16xi1> from vector<8x8x16xi1>
  // CHECK: %[[LHS_MASKED:.+]] = arith.select %[[LHS_MASK]], %[[LHS]], %[[IDENTITY]]
  // CHECK: %[[RHS_MASKED:.+]] = arith.select %[[RHS_MASK]], %[[RHS]], %[[IDENTITY]]
  // CHECK: vector.contract {{.*}} %[[LHS_MASKED]], %[[RHS_MASKED]], %[[ACC]]

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

// Test: contract with constant_mask projects bounds per operand.
// constant_mask [5, 7, 12] with maps (m,k), (k,n), (m,n):
//   LHS mask bounds [5, 12], RHS mask bounds [12, 7].
// CHECK-LABEL: func.func @unwrap_masked_contract_constant_mask(
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<8x16xf32>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<16x8xf32>
// CHECK-SAME: %[[ACC:[a-zA-Z0-9]+]]: vector<8x8xf32>
func.func @unwrap_masked_contract_constant_mask(
    %lhs: vector<8x16xf32>,
    %rhs: vector<16x8xf32>,
    %acc: vector<8x8xf32>) -> vector<8x8xf32> {
  // CHECK-DAG: %[[BOUND_K:.+]] = arith.constant dense<12> : vector<16xindex>
  // CHECK-DAG: %[[BOUND_M:.+]] = arith.constant dense<5> : vector<8xindex>
  // CHECK-DAG: %[[BOUND_N:.+]] = arith.constant dense<7> : vector<8xindex>
  // CHECK-DAG: %[[IDENTITY_LHS:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
  // CHECK-DAG: %[[IDENTITY_RHS:.*]] = arith.constant dense<0.000000e+00> : vector<16x8xf32>
  // CHECK: %[[LHS_MASK:.+]] = arith.andi {{.+}} : vector<8x16xi1>
  // CHECK: %[[RHS_MASK:.+]] = arith.andi {{.+}} : vector<16x8xi1>
  // CHECK: %[[LHS_MASKED:.+]] = arith.select %[[LHS_MASK]], %[[LHS]], %[[IDENTITY_LHS]]
  // CHECK: %[[RHS_MASKED:.+]] = arith.select %[[RHS_MASK]], %[[RHS]], %[[IDENTITY_RHS]]
  // CHECK: vector.contract {{.+}} %[[LHS_MASKED]], %[[RHS_MASKED]], %[[ACC]]

  %mask = vector.constant_mask [5, 7, 12] : vector<8x8x16xi1>
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

// CHECK-LABEL: func.func @unwrap_masked_multi_reduction_add(
// CHECK-SAME: %[[SRC:[a-zA-Z0-9]+]]: vector<16x64xf32>
// CHECK-SAME: %[[ACC:[a-zA-Z0-9]+]]: vector<16xf32>
// CHECK-SAME: %[[DIM:[a-zA-Z0-9]+]]: index
func.func @unwrap_masked_multi_reduction_add(%src: vector<16x64xf32>, %acc: vector<16xf32>, %dim: index) -> vector<16xf32> {
  // CHECK-DAG: %[[IDENTITY:.+]] = arith.constant dense<0.000000e+00> : vector<16x64xf32>
  // CHECK: %[[STEP:.+]] = vector.step : vector<64xindex>
  // CHECK: %[[BCAST_DIM:.+]] = vector.broadcast %[[DIM]] : index to vector<64xindex>
  // CHECK: %[[CMP:.+]] = arith.cmpi slt, %[[STEP]], %[[BCAST_DIM]] : vector<64xindex>
  // CHECK: %[[MASK:.+]] = vector.broadcast %[[CMP]] : vector<64xi1> to vector<16x64xi1>
  // CHECK: %[[MASKED:.+]] = arith.select %[[MASK]], %[[SRC]], %[[IDENTITY]]
  // CHECK: vector.multi_reduction <add>, %[[MASKED]], %[[ACC]] [1] : vector<16x64xf32> to vector<16xf32>

  %c16 = arith.constant 16 : index
  %mask = vector.create_mask %c16, %dim : vector<16x64xi1>
  %result = vector.mask %mask {
    vector.multi_reduction <add>, %src, %acc [1] : vector<16x64xf32> to vector<16xf32>
  } : vector<16x64xi1> -> vector<16xf32>
  return %result : vector<16xf32>
}

// -----

// CHECK-LABEL: func.func @unwrap_masked_multi_reduction_maximumf(
// CHECK-SAME: %[[SRC:[a-zA-Z0-9]+]]: vector<16x64xf32>
// CHECK-SAME: %[[ACC:[a-zA-Z0-9]+]]: vector<16xf32>
// CHECK-SAME: %[[DIM:[a-zA-Z0-9]+]]: index
func.func @unwrap_masked_multi_reduction_maximumf(%src: vector<16x64xf32>, %acc: vector<16xf32>, %dim: index) -> vector<16xf32> {
  // CHECK-DAG: %[[IDENTITY:.+]] = arith.constant dense<0xFF800000> : vector<16x64xf32>
  // CHECK: %[[STEP:.+]] = vector.step : vector<64xindex>
  // CHECK: %[[BCAST_DIM:.+]] = vector.broadcast %[[DIM]] : index to vector<64xindex>
  // CHECK: %[[CMP:.+]] = arith.cmpi slt, %[[STEP]], %[[BCAST_DIM]] : vector<64xindex>
  // CHECK: %[[MASK:.+]] = vector.broadcast %[[CMP]] : vector<64xi1> to vector<16x64xi1>
  // CHECK: %[[MASKED:.+]] = arith.select %[[MASK]], %[[SRC]], %[[IDENTITY]]
  // CHECK: vector.multi_reduction <maximumf>, %[[MASKED]], %[[ACC]] [1] : vector<16x64xf32> to vector<16xf32>

  %c16 = arith.constant 16 : index
  %mask = vector.create_mask %c16, %dim : vector<16x64xi1>
  %result = vector.mask %mask {
    vector.multi_reduction <maximumf>, %src, %acc [1] : vector<16x64xf32> to vector<16xf32>
  } : vector<16x64xi1> -> vector<16xf32>
  return %result : vector<16xf32>
}

// -----

// CHECK-LABEL: func.func @unwrap_masked_multi_reduction_multi_dim(
// CHECK-SAME: %[[SRC:[a-zA-Z0-9]+]]: vector<4x16x64xf32>
// CHECK-SAME: %[[ACC:[a-zA-Z0-9]+]]: vector<4xf32>
// CHECK-SAME: %[[D1:[a-zA-Z0-9]+]]: index
// CHECK-SAME: %[[D2:[a-zA-Z0-9]+]]: index
func.func @unwrap_masked_multi_reduction_multi_dim(%src: vector<4x16x64xf32>, %acc: vector<4xf32>, %d1: index, %d2: index) -> vector<4xf32> {
  // dim 0 has c4 == 4, skipped. dim 1 (%d1) and dim 2 (%d2) are dynamic.
  // CHECK-DAG: %[[IDENTITY:.+]] = arith.constant dense<0.000000e+00> : vector<4x16x64xf32>
  // CHECK: %[[STEP1:.+]] = vector.step : vector<16xindex>
  // CHECK: %[[BCAST_D1:.+]] = vector.broadcast %[[D1]] : index to vector<16xindex>
  // CHECK: %[[CMP1:.+]] = arith.cmpi slt, %[[STEP1]], %[[BCAST_D1]] : vector<16xindex>
  // CHECK: %[[BCAST_CMP1:.+]] = vector.broadcast %[[CMP1]] : vector<16xi1> to vector<4x64x16xi1>
  // CHECK: %[[TRANS1:.+]] = vector.transpose %[[BCAST_CMP1]], [0, 2, 1] : vector<4x64x16xi1> to vector<4x16x64xi1>
  // CHECK: %[[STEP2:.+]] = vector.step : vector<64xindex>
  // CHECK: %[[BCAST_D2:.+]] = vector.broadcast %[[D2]] : index to vector<64xindex>
  // CHECK: %[[CMP2:.+]] = arith.cmpi slt, %[[STEP2]], %[[BCAST_D2]] : vector<64xindex>
  // CHECK: %[[BCAST_CMP2:.+]] = vector.broadcast %[[CMP2]] : vector<64xi1> to vector<4x16x64xi1>
  // CHECK: %[[MASK:.+]] = arith.andi %[[TRANS1]], %[[BCAST_CMP2]] : vector<4x16x64xi1>
  // CHECK: %[[MASKED:.+]] = arith.select %[[MASK]], %[[SRC]], %[[IDENTITY]]
  // CHECK: vector.multi_reduction <add>, %[[MASKED]], %[[ACC]] [1, 2] : vector<4x16x64xf32> to vector<4xf32>

  %c4 = arith.constant 4 : index
  %mask = vector.create_mask %c4, %d1, %d2 : vector<4x16x64xi1>
  %result = vector.mask %mask {
    vector.multi_reduction <add>, %src, %acc [1, 2] : vector<4x16x64xf32> to vector<4xf32>
  } : vector<4x16x64xi1> -> vector<4xf32>
  return %result : vector<4xf32>
}

// -----

func.func @masked_reduction_add(%src : vector<16xf32>, %mask : vector<16xi1>) -> f32 {
  // CHECK-LABEL: func @masked_reduction_add
  // CHECK-SAME: %[[SRC:[a-zA-Z0-9]+]]: vector<16xf32>
  // CHECK-SAME: %[[MASK:[a-zA-Z0-9]+]]: vector<16xi1>
  // CHECK-DAG: %[[IDENTITY:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
  // CHECK: %[[MASKED_SRC:.+]] = arith.select %[[MASK]], %[[SRC]], %[[IDENTITY]]
  // CHECK: %[[RESULT:.+]] = vector.reduction <add>, %[[MASKED_SRC]] : vector<16xf32> into f32

  // CHECK: return %[[RESULT]]
  %result = vector.mask %mask {
    vector.reduction <add>, %src : vector<16xf32> into f32
  } : vector<16xi1> -> f32
  return %result : f32
}

// -----

func.func @masked_reduction_add_with_acc(%src : vector<16xf32>, %acc : f32, %mask : vector<16xi1>) -> f32 {
  // CHECK-LABEL: func @masked_reduction_add_with_acc
  // CHECK-SAME: %[[SRC:[a-zA-Z0-9]+]]: vector<16xf32>
  // CHECK-SAME: %[[ACC:[a-zA-Z0-9]+]]: f32
  // CHECK-SAME: %[[MASK:[a-zA-Z0-9]+]]: vector<16xi1>
  // CHECK-DAG: %[[IDENTITY:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
  // CHECK: %[[MASKED_SRC:.+]] = arith.select %[[MASK]], %[[SRC]], %[[IDENTITY]]
  // CHECK: %[[RESULT:.+]] = vector.reduction <add>, %[[MASKED_SRC]], %[[ACC]] : vector<16xf32> into f32

  // CHECK: return %[[RESULT]]
  %result = vector.mask %mask {
    vector.reduction <add>, %src, %acc : vector<16xf32> into f32
  } : vector<16xi1> -> f32
  return %result : f32
}

// -----

func.func @masked_reduction_maximumf(%src : vector<16xf32>, %mask : vector<16xi1>) -> f32 {
  // CHECK-LABEL: func @masked_reduction_maximumf
  // CHECK-SAME: %[[SRC:[a-zA-Z0-9]+]]: vector<16xf32>
  // CHECK-SAME: %[[MASK:[a-zA-Z0-9]+]]: vector<16xi1>
  // CHECK-DAG: %[[IDENTITY:.+]] = arith.constant dense<0xFF800000> : vector<16xf32>
  // CHECK: %[[MASKED_SRC:.+]] = arith.select %[[MASK]], %[[SRC]], %[[IDENTITY]]
  // CHECK: %[[RESULT:.+]] = vector.reduction <maximumf>, %[[MASKED_SRC]] : vector<16xf32> into f32

  // CHECK: return %[[RESULT]]
  %result = vector.mask %mask {
    vector.reduction <maximumf>, %src : vector<16xf32> into f32
  } : vector<16xi1> -> f32
  return %result : f32
}

// -----

// CHECK-LABEL: func.func @decompose_constant_mask
func.func @decompose_constant_mask() -> vector<256x128xi1> {
  // dim 0: 17 != 256, decomposed. dim 1: 128 == 128, skipped.
  // The constant bound is folded into a splat vector constant.
  // CHECK-DAG: %[[BCAST:.+]] = arith.constant dense<17> : vector<256xindex>
  // CHECK: %[[STEP:.+]] = vector.step : vector<256xindex>
  // CHECK: %[[CMP:.+]] = arith.cmpi slt, %[[STEP]], %[[BCAST]] : vector<256xindex>
  // CHECK: %[[BCAST_CMP:.+]] = vector.broadcast %[[CMP]] : vector<256xi1> to vector<128x256xi1>
  // CHECK: %[[RESULT:.+]] = vector.transpose %[[BCAST_CMP]], [1, 0] : vector<128x256xi1> to vector<256x128xi1>
  // CHECK: return %[[RESULT]]
  %mask = vector.constant_mask [17, 128] : vector<256x128xi1>
  return %mask : vector<256x128xi1>
}

// -----

// CHECK-LABEL: func.func @decompose_create_mask_1d(
// CHECK-SAME: %[[N:[a-zA-Z0-9]+]]: index
func.func @decompose_create_mask_1d(%n: index) -> vector<64xi1> {
  // CHECK: %[[STEP:.+]] = vector.step : vector<64xindex>
  // CHECK: %[[BCAST:.+]] = vector.broadcast %[[N]] : index to vector<64xindex>
  // CHECK: %[[CMP:.+]] = arith.cmpi slt, %[[STEP]], %[[BCAST]] : vector<64xindex>
  // CHECK: return %[[CMP]]
  %mask = vector.create_mask %n : vector<64xi1>
  return %mask : vector<64xi1>
}

// -----

// Test: all-true constant_mask (all bounds equal dim sizes) produces a constant.
// CHECK-LABEL: func.func @decompose_all_true_constant_mask
func.func @decompose_all_true_constant_mask() -> vector<8x16xi1> {
  // CHECK: %[[RESULT:.+]] = arith.constant dense<true> : vector<8x16xi1>
  // CHECK: return %[[RESULT]]
  %mask = vector.constant_mask [8, 16] : vector<8x16xi1>
  return %mask : vector<8x16xi1>
}

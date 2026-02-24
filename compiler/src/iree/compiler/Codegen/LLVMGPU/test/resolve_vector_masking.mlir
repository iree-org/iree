// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-llvmgpu-resolve-vector-masking))" %s | FileCheck %s --implicit-check-not="vector.mask"

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
  // CHECK-DAG: %[[IDENTITY_LHS:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
  // CHECK-DAG: %[[IDENTITY_RHS:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf32>
  // CHECK: %[[LHS_T:.+]] = vector.transpose %[[MASK]], [1, 0, 2] : vector<8x8x16xi1> to vector<8x8x16xi1>
  // CHECK: %[[LHS_MASK:.+]] = vector.extract %[[LHS_T]][0] : vector<8x16xi1> from vector<8x8x16xi1>
  // CHECK: %[[RHS_T:.+]] = vector.transpose %[[MASK]], [0, 1, 2] : vector<8x8x16xi1> to vector<8x8x16xi1>
  // CHECK: %[[RHS_MASK:.+]] = vector.extract %[[RHS_T]][0] : vector<8x16xi1> from vector<8x8x16xi1>
  // CHECK: %[[LHS_MASKED:.+]] = arith.select %[[LHS_MASK]], %[[LHS]], %[[IDENTITY_LHS]]
  // CHECK: %[[RHS_MASKED:.+]] = arith.select %[[RHS_MASK]], %[[RHS]], %[[IDENTITY_RHS]]
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

// CHECK-LABEL: func.func @unwrap_masked_multi_reduction_add(
// CHECK-SAME: %[[SRC:[a-zA-Z0-9]+]]: vector<16x64xf32>
// CHECK-SAME: %[[ACC:[a-zA-Z0-9]+]]: vector<16xf32>
// CHECK-SAME: %[[DIM:[a-zA-Z0-9]+]]: index
func.func @unwrap_masked_multi_reduction_add(%src: vector<16x64xf32>, %acc: vector<16xf32>, %dim: index) -> vector<16xf32> {
  // CHECK-DAG: %[[IDENTITY:.+]] = arith.constant dense<0.000000e+00> : vector<16x64xf32>
  // CHECK-DAG: %[[MASK:.+]] = vector.create_mask %{{.+}}, %[[DIM]]
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
  // CHECK-DAG: %[[MASK:.+]] = vector.create_mask %{{.+}}, %[[DIM]]
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

// CHECK-LABEL: func.func @preserve_attributes_multi_reduction
func.func @preserve_attributes_multi_reduction(%src: vector<16x64xf32>, %acc: vector<16xf32>, %dim: index) -> vector<16xf32> {
  // CHECK: arith.select
  // CHECK: %{{.+}} = vector.multi_reduction <add>
  // CHECK-SAME: {iree.test.attr = "preserved"}

  %c16 = arith.constant 16 : index
  %mask = vector.create_mask %c16, %dim : vector<16x64xi1>
  %result = vector.mask %mask {
    vector.multi_reduction <add>, %src, %acc {iree.test.attr = "preserved"} [1] : vector<16x64xf32> to vector<16xf32>
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
  // CHECK-DAG: %[[IDENTITY:.+]] = arith.constant dense<0.000000e+00> : vector<4x16x64xf32>
  // CHECK-DAG: %[[MASK:.+]] = vector.create_mask %{{.+}}, %[[D1]], %[[D2]]
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

func.func @masked_reduction_with_attr(%src : vector<16xf32>, %mask : vector<16xi1>) -> f32 {
  // CHECK-LABEL: func @masked_reduction_with_attr
  // CHECK: %{{.+}} = vector.reduction <add>
  // CHECK-SAME: {iree.test.attr = "preserved"}

  %result = vector.mask %mask {
    vector.reduction <add>, %src {iree.test.attr = "preserved"} : vector<16xf32> into f32
  } : vector<16xi1> -> f32
  return %result : f32
}

// RUN: iree-opt --iree-llvmcpu-vector-contract-custom-kernels %s | FileCheck %s

// CHECK-LABEL: @vector_i8i8i32matmul(
// CHECK-SAME:          %[[LHS:[^:[:space:]]+]]
// CHECK-SAME:          %[[RHS:[^:[:space:]]+]]
// CHECK-SAME:          %[[ACC:[^:[:space:]]+]]
// CHECK-DAG:       %[[ZERO:.*]]          = arith.constant dense<0> : vector<4x4xi8>
// CHECK-DAG:       %[[ACC_ROW_0:.*]]     = vector.extract %[[ACC]][0] : vector<8xi32> from vector<8x8xi32>
// CHECK-DAG:       %[[ACC_ROW_1:.*]]     = vector.extract %[[ACC]][1] : vector<8xi32> from vector<8x8xi32>
// CHECK-DAG:       %[[ACC_ROW_2:.*]]     = vector.extract %[[ACC]][2] : vector<8xi32> from vector<8x8xi32>
// CHECK-DAG:       %[[ACC_ROW_3:.*]]     = vector.extract %[[ACC]][3] : vector<8xi32> from vector<8x8xi32>
// CHECK-DAG:       %[[ACC_ROW_4:.*]]     = vector.extract %[[ACC]][4] : vector<8xi32> from vector<8x8xi32>
// CHECK-DAG:       %[[ACC_ROW_5:.*]]     = vector.extract %[[ACC]][5] : vector<8xi32> from vector<8x8xi32>
// CHECK-DAG:       %[[ACC_ROW_6:.*]]     = vector.extract %[[ACC]][6] : vector<8xi32> from vector<8x8xi32>
// CHECK-DAG:       %[[ACC_ROW_7:.*]]     = vector.extract %[[ACC]][7] : vector<8xi32> from vector<8x8xi32>
// CHECK-DAG:       %[[ACC_CHUNK_00:.*]]  = vector.extract_strided_slice %[[ACC_ROW_0]] {offsets = [0]
// CHECK-DAG:       %[[ACC_CHUNK_01:.*]]  = vector.extract_strided_slice %[[ACC_ROW_0]] {offsets = [4]
// CHECK-DAG:       %[[ACC_CHUNK_02:.*]]  = vector.extract_strided_slice %[[ACC_ROW_1]] {offsets = [0]
// CHECK-DAG:       %[[ACC_CHUNK_03:.*]]  = vector.extract_strided_slice %[[ACC_ROW_1]] {offsets = [4]
// CHECK-DAG:       %[[ACC_CHUNK_04:.*]]  = vector.extract_strided_slice %[[ACC_ROW_2]] {offsets = [0]
// CHECK-DAG:       %[[ACC_CHUNK_05:.*]]  = vector.extract_strided_slice %[[ACC_ROW_2]] {offsets = [4]
// CHECK-DAG:       %[[ACC_CHUNK_06:.*]]  = vector.extract_strided_slice %[[ACC_ROW_3]] {offsets = [0]
// CHECK-DAG:       %[[ACC_CHUNK_07:.*]]  = vector.extract_strided_slice %[[ACC_ROW_3]] {offsets = [4]
// CHECK-DAG:       %[[ACC_CHUNK_08:.*]]  = vector.extract_strided_slice %[[ACC_ROW_4]] {offsets = [0]
// CHECK-DAG:       %[[ACC_CHUNK_09:.*]]  = vector.extract_strided_slice %[[ACC_ROW_4]] {offsets = [4]
// CHECK-DAG:       %[[ACC_CHUNK_10:.*]]  = vector.extract_strided_slice %[[ACC_ROW_5]] {offsets = [0]
// CHECK-DAG:       %[[ACC_CHUNK_11:.*]]  = vector.extract_strided_slice %[[ACC_ROW_5]] {offsets = [4]
// CHECK-DAG:       %[[ACC_CHUNK_12:.*]]  = vector.extract_strided_slice %[[ACC_ROW_6]] {offsets = [0]
// CHECK-DAG:       %[[ACC_CHUNK_13:.*]]  = vector.extract_strided_slice %[[ACC_ROW_6]] {offsets = [4]
// CHECK-DAG:       %[[ACC_CHUNK_14:.*]]  = vector.extract_strided_slice %[[ACC_ROW_7]] {offsets = [0]
// CHECK-DAG:       %[[ACC_CHUNK_15:.*]]  = vector.extract_strided_slice %[[ACC_ROW_7]] {offsets = [4]
// CHECK-DAG:       %[[LHS_HALF_0:.*]]    = vector.extract_strided_slice %[[LHS]] {offsets = [0, 0]
// CHECK-DAG:       %[[LHS_HALF_1:.*]]    = vector.extract_strided_slice %[[LHS]] {offsets = [4, 0]
// CHECK-DAG:       %[[RHS_HALF_0:.*]]    = vector.extract_strided_slice %[[RHS]] {offsets = [0, 0]
// CHECK-DAG:       %[[RHS_HALF_1:.*]]    = vector.extract_strided_slice %[[RHS]] {offsets = [4, 0]
// CHECK-DAG:       %[[LHS_CHUNK_00:.*]]  = vector.shuffle %[[LHS_HALF_0]], %[[ZERO]] [0, 0, 0, 0]
// CHECK-DAG:       %[[LHS_CHUNK_01:.*]]  = vector.shuffle %[[LHS_HALF_0]], %[[ZERO]] [0, 0, 0, 0]
// CHECK-DAG:       %[[LHS_CHUNK_02:.*]]  = vector.shuffle %[[LHS_HALF_0]], %[[ZERO]] [1, 1, 1, 1]
// CHECK-DAG:       %[[LHS_CHUNK_03:.*]]  = vector.shuffle %[[LHS_HALF_0]], %[[ZERO]] [1, 1, 1, 1]
// CHECK-DAG:       %[[LHS_CHUNK_04:.*]]  = vector.shuffle %[[LHS_HALF_0]], %[[ZERO]] [2, 2, 2, 2]
// CHECK-DAG:       %[[LHS_CHUNK_05:.*]]  = vector.shuffle %[[LHS_HALF_0]], %[[ZERO]] [2, 2, 2, 2]
// CHECK-DAG:       %[[LHS_CHUNK_06:.*]]  = vector.shuffle %[[LHS_HALF_0]], %[[ZERO]] [3, 3, 3, 3]
// CHECK-DAG:       %[[LHS_CHUNK_07:.*]]  = vector.shuffle %[[LHS_HALF_0]], %[[ZERO]] [3, 3, 3, 3]
// CHECK-DAG:       %[[LHS_CHUNK_08:.*]]  = vector.shuffle %[[LHS_HALF_1]], %[[ZERO]] [0, 0, 0, 0]
// CHECK-DAG:       %[[LHS_CHUNK_09:.*]]  = vector.shuffle %[[LHS_HALF_1]], %[[ZERO]] [0, 0, 0, 0]
// CHECK-DAG:       %[[LHS_CHUNK_10:.*]]  = vector.shuffle %[[LHS_HALF_1]], %[[ZERO]] [1, 1, 1, 1]
// CHECK-DAG:       %[[LHS_CHUNK_11:.*]]  = vector.shuffle %[[LHS_HALF_1]], %[[ZERO]] [1, 1, 1, 1]
// CHECK-DAG:       %[[LHS_CHUNK_12:.*]]  = vector.shuffle %[[LHS_HALF_1]], %[[ZERO]] [2, 2, 2, 2]
// CHECK-DAG:       %[[LHS_CHUNK_13:.*]]  = vector.shuffle %[[LHS_HALF_1]], %[[ZERO]] [2, 2, 2, 2]
// CHECK-DAG:       %[[LHS_CHUNK_14:.*]]  = vector.shuffle %[[LHS_HALF_1]], %[[ZERO]] [3, 3, 3, 3]
// CHECK-DAG:       %[[LHS_CHUNK_15:.*]]  = vector.shuffle %[[LHS_HALF_1]], %[[ZERO]] [3, 3, 3, 3]
// CHECK-DAG:       %[[SDOT_CHUNK_00:.*]] = arm_neon.2d.sdot %[[ACC_CHUNK_00]], %[[RHS_HALF_0]], %[[LHS_CHUNK_00]]
// CHECK-DAG:       %[[SDOT_CHUNK_01:.*]] = arm_neon.2d.sdot %[[ACC_CHUNK_01]], %[[RHS_HALF_1]], %[[LHS_CHUNK_01]]
// CHECK-DAG:       %[[SDOT_CHUNK_02:.*]] = arm_neon.2d.sdot %[[ACC_CHUNK_02]], %[[RHS_HALF_0]], %[[LHS_CHUNK_02]]
// CHECK-DAG:       %[[SDOT_CHUNK_03:.*]] = arm_neon.2d.sdot %[[ACC_CHUNK_03]], %[[RHS_HALF_1]], %[[LHS_CHUNK_03]]
// CHECK-DAG:       %[[SDOT_CHUNK_04:.*]] = arm_neon.2d.sdot %[[ACC_CHUNK_04]], %[[RHS_HALF_0]], %[[LHS_CHUNK_04]]
// CHECK-DAG:       %[[SDOT_CHUNK_05:.*]] = arm_neon.2d.sdot %[[ACC_CHUNK_05]], %[[RHS_HALF_1]], %[[LHS_CHUNK_05]]
// CHECK-DAG:       %[[SDOT_CHUNK_06:.*]] = arm_neon.2d.sdot %[[ACC_CHUNK_06]], %[[RHS_HALF_0]], %[[LHS_CHUNK_06]]
// CHECK-DAG:       %[[SDOT_CHUNK_07:.*]] = arm_neon.2d.sdot %[[ACC_CHUNK_07]], %[[RHS_HALF_1]], %[[LHS_CHUNK_07]]
// CHECK-DAG:       %[[SDOT_CHUNK_08:.*]] = arm_neon.2d.sdot %[[ACC_CHUNK_08]], %[[RHS_HALF_0]], %[[LHS_CHUNK_08]]
// CHECK-DAG:       %[[SDOT_CHUNK_09:.*]] = arm_neon.2d.sdot %[[ACC_CHUNK_09]], %[[RHS_HALF_1]], %[[LHS_CHUNK_09]]
// CHECK-DAG:       %[[SDOT_CHUNK_10:.*]] = arm_neon.2d.sdot %[[ACC_CHUNK_10]], %[[RHS_HALF_0]], %[[LHS_CHUNK_10]]
// CHECK-DAG:       %[[SDOT_CHUNK_11:.*]] = arm_neon.2d.sdot %[[ACC_CHUNK_11]], %[[RHS_HALF_1]], %[[LHS_CHUNK_11]]
// CHECK-DAG:       %[[SDOT_CHUNK_12:.*]] = arm_neon.2d.sdot %[[ACC_CHUNK_12]], %[[RHS_HALF_0]], %[[LHS_CHUNK_12]]
// CHECK-DAG:       %[[SDOT_CHUNK_13:.*]] = arm_neon.2d.sdot %[[ACC_CHUNK_13]], %[[RHS_HALF_1]], %[[LHS_CHUNK_13]]
// CHECK-DAG:       %[[SDOT_CHUNK_14:.*]] = arm_neon.2d.sdot %[[ACC_CHUNK_14]], %[[RHS_HALF_0]], %[[LHS_CHUNK_14]]
// CHECK-DAG:       %[[SDOT_CHUNK_15:.*]] = arm_neon.2d.sdot %[[ACC_CHUNK_15]], %[[RHS_HALF_1]], %[[LHS_CHUNK_15]]
// CHECK-DAG:       %[[RES_00:.*]]        = vector.insert_strided_slice %[[SDOT_CHUNK_00]], %[[ACC]]    {offsets = [0, 0]
// CHECK-DAG:       %[[RES_01:.*]]        = vector.insert_strided_slice %[[SDOT_CHUNK_01]], %[[RES_00]] {offsets = [0, 4]
// CHECK-DAG:       %[[RES_02:.*]]        = vector.insert_strided_slice %[[SDOT_CHUNK_02]], %[[RES_01]] {offsets = [1, 0]
// CHECK-DAG:       %[[RES_03:.*]]        = vector.insert_strided_slice %[[SDOT_CHUNK_03]], %[[RES_02]] {offsets = [1, 4]
// CHECK-DAG:       %[[RES_04:.*]]        = vector.insert_strided_slice %[[SDOT_CHUNK_04]], %[[RES_03]] {offsets = [2, 0]
// CHECK-DAG:       %[[RES_05:.*]]        = vector.insert_strided_slice %[[SDOT_CHUNK_05]], %[[RES_04]] {offsets = [2, 4]
// CHECK-DAG:       %[[RES_06:.*]]        = vector.insert_strided_slice %[[SDOT_CHUNK_06]], %[[RES_05]] {offsets = [3, 0]
// CHECK-DAG:       %[[RES_07:.*]]        = vector.insert_strided_slice %[[SDOT_CHUNK_07]], %[[RES_06]] {offsets = [3, 4]
// CHECK-DAG:       %[[RES_08:.*]]        = vector.insert_strided_slice %[[SDOT_CHUNK_08]], %[[RES_07]] {offsets = [4, 0]
// CHECK-DAG:       %[[RES_09:.*]]        = vector.insert_strided_slice %[[SDOT_CHUNK_09]], %[[RES_08]] {offsets = [4, 4]
// CHECK-DAG:       %[[RES_10:.*]]        = vector.insert_strided_slice %[[SDOT_CHUNK_10]], %[[RES_09]] {offsets = [5, 0]
// CHECK-DAG:       %[[RES_11:.*]]        = vector.insert_strided_slice %[[SDOT_CHUNK_11]], %[[RES_10]] {offsets = [5, 4]
// CHECK-DAG:       %[[RES_12:.*]]        = vector.insert_strided_slice %[[SDOT_CHUNK_12]], %[[RES_11]] {offsets = [6, 0]
// CHECK-DAG:       %[[RES_13:.*]]        = vector.insert_strided_slice %[[SDOT_CHUNK_13]], %[[RES_12]] {offsets = [6, 4]
// CHECK-DAG:       %[[RES_14:.*]]        = vector.insert_strided_slice %[[SDOT_CHUNK_14]], %[[RES_13]] {offsets = [7, 0]
// CHECK-DAG:       %[[RES_15:.*]]        = vector.insert_strided_slice %[[SDOT_CHUNK_15]], %[[RES_14]] {offsets = [7, 4]
// CHECK:           return %[[RES_15]]
func.func @vector_i8i8i32matmul(
    %lhs: vector<8x4xi8>,
    %rhs: vector<8x4xi8>,
    %acc: vector<8x8xi32>) -> vector<8x8xi32> attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod", prefer_intrinsics_over_asm=true}>
} {
  %lhs_wide = arith.extsi %lhs : vector<8x4xi8> to vector<8x4xi32>
  %rhs_wide = arith.extsi %rhs : vector<8x4xi8> to vector<8x4xi32>
  %res = vector.contract {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
      ], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
  } %lhs_wide, %rhs_wide, %acc : vector<8x4xi32>, vector<8x4xi32> into vector<8x8xi32>
  return %res : vector<8x8xi32>
}

// -----

// CHECK-LABEL: @vector_f32f32f32matmul(
func.func @vector_f32f32f32matmul(
    %lhs: vector<8x4xf32>,
    %rhs: vector<8x4xf32>,
    %acc: vector<8x8xf32>) -> vector<8x8xf32> attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod", prefer_intrinsics_over_asm=true}>
} {
  // CHECK: vector.contract
  %res = vector.contract {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
      ], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
  } %lhs, %rhs, %acc : vector<8x4xf32>, vector<8x4xf32> into vector<8x8xf32>
  return %res : vector<8x8xf32>
}


// -----

// CHECK-LABEL: @vector_i32i32i32matmul(
func.func @vector_i32i32i32matmul(
    %lhs: vector<8x4xi32>,
    %rhs: vector<8x4xi32>,
    %acc: vector<8x8xi32>) -> vector<8x8xi32> attributes {
  hal.executable.target = #hal.executable.target<"xyz", "xyz", {target_triple="aarch64-xyz-xyz", cpu_features="+dotprod", prefer_intrinsics_over_asm=true}>
} {
  // CHECK: vector.contract
  %res = vector.contract {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
      ], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
  } %lhs, %rhs, %acc : vector<8x4xi32>, vector<8x4xi32> into vector<8x8xi32>
  return %res : vector<8x8xi32>
}

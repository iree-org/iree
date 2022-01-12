// RUN: iree-opt -iree-llvmcpu-vector-to-aarch64-inline-asm %s | FileCheck %s

// CHECK-LABEL: @vector_i8i8i32matmul_to_aarch64_asm_vec_dot(
func @vector_i8i8i32matmul_to_aarch64_asm_vec_dot(
    // CHECK-SAME: %[[LHS:[a-zA-Z0-9_]+]]
    %lhs: vector<4x4xi8>,
    // CHECK-SAME: %[[RHS:[a-zA-Z0-9_]+]]
    %rhs: vector<4x4xi8>,
    // CHECK-SAME: %[[ACC:[a-zA-Z0-9_]+]]
    %acc: vector<4x4xi32>) -> vector<4x4xi32> {
  %lhs_wide = arith.extsi %lhs : vector<4x4xi8> to vector<4x4xi32>
  %rhs_wide = arith.extsi %rhs : vector<4x4xi8> to vector<4x4xi32>
  // CHECK-DAG: %[[RES_2D:.+]] = arith.constant dense<0> : vector<4x4xi32>
  // CHECK-DAG: %[[DST0:.+]] = vector.extract %[[ACC]][0] : vector<4x4xi32>
  // CHECK-DAG: %[[DST1:.+]] = vector.extract %[[ACC]][1] : vector<4x4xi32>
  // CHECK-DAG: %[[DST2:.+]] = vector.extract %[[ACC]][2] : vector<4x4xi32>
  // CHECK-DAG: %[[DST3:.+]] = vector.extract %[[ACC]][3] : vector<4x4xi32>
  // CHECK-DAG: %[[LHS_1D:.+]] = vector.shape_cast %[[LHS]] : vector<4x4xi8> to vector<16xi8>
  // CHECK-DAG: %[[RHS_T_2d:.+]] = vector.transpose %[[RHS]], [1, 0]
  // CHECK-DAG: %[[RHS_T:.+]] = vector.shape_cast %[[RHS_T_2d]] : vector<4x4xi8> to vector<16xi8>
  //     CHECK: %[[ASM_RESULT:.+]] = llvm.inline_asm {{.*}} "=w,=w,=w,=w,w,w,0,1,2,3" %[[LHS_1D]], %[[RHS_T]], %[[DST0]], %[[DST1]], %[[DST2]], %[[DST3]]
  // CHECK-DAG: %[[RES_0:.+]] = llvm.extractvalue %[[ASM_RESULT]][0] : !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>)>
  // CHECK-DAG: %[[RES_1:.+]] = llvm.extractvalue %[[ASM_RESULT]][1] : !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>)>
  // CHECK-DAG: %[[RES_2:.+]] = llvm.extractvalue %[[ASM_RESULT]][2] : !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>)>
  // CHECK-DAG: %[[RES_3:.+]] = llvm.extractvalue %[[ASM_RESULT]][3] : !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>)>
  // CHECK-DAG: %[[RES_2D_0:.+]] = vector.insert %[[RES_0]], %[[RES_2D]] [0] : vector<4xi32> into vector<4x4xi32>
  // CHECK-DAG: %[[RES_2D_1:.+]] = vector.insert %[[RES_1]], %[[RES_2D_0]] [1] : vector<4xi32> into vector<4x4xi32>
  // CHECK-DAG: %[[RES_2D_2:.+]] = vector.insert %[[RES_2]], %[[RES_2D_1]] [2] : vector<4xi32> into vector<4x4xi32>
  // CHECK-DAG: %[[RES_2D_3:.+]] = vector.insert %[[RES_3]], %[[RES_2D_2]] [3] : vector<4xi32> into vector<4x4xi32>
  %res = vector.contract {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d2, d1)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
      ], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
  } %lhs_wide, %rhs_wide, %acc : vector<4x4xi32>, vector<4x4xi32> into vector<4x4xi32>
  // CHECK: return %[[RES_2D_3]]
  return %res : vector<4x4xi32>
}

// RUN: iree-opt -iree-llvmcpu-vector-contract-custom-kernels='aarch64 dotprod intrinsics' %s | FileCheck %s

// CHECK-LABEL: @vector_i8i8i32matmul_to_aarch64_asm_vec_dot(
func @vector_i8i8i32matmul_to_aarch64_asm_vec_dot(
    // CHECK-SAME: %[[LHS:[a-zA-Z0-9_]+]]
    %lhs: vector<8x4xi8>,
    // CHECK-SAME: %[[RHS:[a-zA-Z0-9_]+]]
    %rhs: vector<8x4xi8>,
    // CHECK-SAME: %[[ACC:[a-zA-Z0-9_]+]]
    %acc: vector<8x8xi32>) -> vector<8x8xi32> {
  %lhs_wide = arith.extsi %lhs : vector<8x4xi8> to vector<8x4xi32>
  %rhs_wide = arith.extsi %rhs : vector<8x4xi8> to vector<8x4xi32>
  // CHECK-NOT: vector.contract
  %res = vector.contract {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
      ], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
  } %lhs_wide, %rhs_wide, %acc : vector<8x4xi32>, vector<8x4xi32> into vector<8x8xi32>
  return %res : vector<8x8xi32>
}

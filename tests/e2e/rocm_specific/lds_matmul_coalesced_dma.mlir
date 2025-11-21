// Test matmul with coalesced DMA where innermost dim == subgroup_size.
// Coalesced DMA requires the innermost dimension to be at least subgroup_size (64 for CDNA)
// so that each lane gets at least one element.

!A_type = tensor<32x64xf32>
!B_type = tensor<64x32xf32>
!C_type = tensor<32x32xf32>

func.func @matmul_32x64x32_f32() {
  %cst = arith.constant 0.000000e+00 : f32
  %c1_i32 = arith.constant 1 : i32

  // Generate lhs where lhs[i,k] = k + 1 (column index + 1)
  %empty_lhs = tensor.empty() : !A_type
  %lhs = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    outs(%empty_lhs : !A_type) {
  ^bb0(%out: f32):
    %k = linalg.index 1 : index
    %k_i32 = arith.index_cast %k : index to i32
    %k_plus_1 = arith.addi %k_i32, %c1_i32 : i32
    %val = arith.sitofp %k_plus_1 : i32 to f32
    linalg.yield %val : f32
  } -> !A_type
  %lhs_unfoldable = util.optimization_barrier %lhs : !A_type

  %rhs = util.unfoldable_constant dense<1.0> : !B_type
  %empty = tensor.empty() : !C_type
  %C = linalg.fill ins(%cst : f32) outs(%empty : !C_type) -> !C_type
  %result = linalg.matmul ins(%lhs_unfoldable, %rhs : !A_type, !B_type)
                          outs(%C : !C_type) -> !C_type
  // lhs[i,k] = k+1, rhs = 1.0
  // result[i,j] = sum_k((k+1) * 1.0) = 1+2+...+64 = 64*65/2 = 2080
  check.expect_almost_eq_const(%result, dense<2080.0> : !C_type) : !C_type
  return
}

!A2_type = tensor<32x64xf32>
!B2_type = tensor<64x64xf32>
!C2_type = tensor<32x64xf32>

func.func @matmul_32x64x64_f32() {
  %cst = arith.constant 0.000000e+00 : f32
  %c1_i32 = arith.constant 1 : i32

  // Generate lhs where lhs[i,k] = k + 1 (column index + 1)
  %empty_lhs = tensor.empty() : !A2_type
  %lhs = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    outs(%empty_lhs : !A2_type) {
  ^bb0(%out: f32):
    %k = linalg.index 1 : index
    %k_i32 = arith.index_cast %k : index to i32
    %k_plus_1 = arith.addi %k_i32, %c1_i32 : i32
    %val = arith.sitofp %k_plus_1 : i32 to f32
    linalg.yield %val : f32
  } -> !A2_type
  %lhs_unfoldable = util.optimization_barrier %lhs : !A2_type

  %rhs = util.unfoldable_constant dense<1.0> : !B2_type
  %empty = tensor.empty() : !C2_type
  %C = linalg.fill ins(%cst : f32) outs(%empty : !C2_type) -> !C2_type
  %result = linalg.matmul ins(%lhs_unfoldable, %rhs : !A2_type, !B2_type)
                          outs(%C : !C2_type) -> !C2_type
  // lhs[i,k] = k+1, rhs = 1.0
  // result[i,j] = sum_k((k+1) * 1.0) = 1+2+...+64 = 64*65/2 = 2080
  check.expect_almost_eq_const(%result, dense<2080.0> : !C2_type) : !C2_type
  return
}

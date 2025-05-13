// Test matmuls of specific sizes using global load lds option for GPU.

!A_size = tensor<16x64xi8>
!B_size = tensor<64x16xi8>
!C_size = tensor<16x16xi32>
!out_size = tensor<16x16xi8>

func.func @matmul_16x64x16_i8() {
  %lhs = util.unfoldable_constant dense<1> : !A_size
  %rhs = util.unfoldable_constant dense<1> : !B_size
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : !C_size
  %C = linalg.fill ins(%cst : f32) outs(%empty : !C_size) -> !C_size
  %0 = linalg.matmul ins(%lhs, %rhs : !A_size, !B_size)
                     outs(%C : !C_size) -> !C_size
  %E = tensor.empty() : !out_size
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
  ins(%0 : !C_size) outs(%E : !out_size) {
  ^bb0(%in: i32, %out: i8):
    %trunc = arith.trunci %in : i32 to i8
    linalg.yield %trunc : i8
  } -> !out_size
  check.expect_eq_const(%1, dense<64> : !out_size) : !out_size
  return
}

// Test matmuls of specific sizes using global load lds option for GPU.

func.func @matmul_16x64x16_i8() {
  %lhs = util.unfoldable_constant dense<1> : tensor<16x64xi8>
  %rhs = util.unfoldable_constant dense<1> : tensor<64x16xi8>
  %c0 = arith.constant 0 : i8
  %init = tensor.empty() : tensor<16x16xi8>
  %CC = linalg.fill ins(%c0 : i8) outs(%init : tensor<16x16xi8>) -> tensor<16x16xi8>
  %D = linalg.matmul ins(%lhs, %rhs: tensor<16x64xi8>, tensor<64x16xi8>)
                    outs(%CC: tensor<16x16xi8>) -> tensor<16x16xi8>
  check.expect_eq_const(%D, dense<64> : tensor<16x16xi8>) : tensor<16x16xi8>
  return
}

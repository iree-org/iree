func.func @test_matmul_i8() {
  %lhs = util.unfoldable_constant dense<123> : tensor<100x100xi8>
  %rhs = util.unfoldable_constant dense<45> : tensor<100x100xi8>
  %init_acc = util.unfoldable_constant dense<6789> : tensor<100x100xi32>
  %result = linalg.matmul
    ins(%lhs, %rhs : tensor<100x100xi8>, tensor<100x100xi8>)
    outs(%init_acc : tensor<100x100xi32>) -> tensor<100x100xi32>
  check.expect_eq_const(%result, dense<560289> : tensor<100x100xi32>) : tensor<100x100xi32>
  return
}

builtin.module @module_b {
  // Note the iree.abi.output indicating that the %output argument is storage
  // for function result 0.
  func.func @mul(%lhs: tensor<4096xf32>, %rhs: tensor<4096xf32>,
                 %output: tensor<4096xf32> {iree.abi.output = 0 : index}) -> tensor<4096xf32> {
    %result = arith.mulf %lhs, %rhs : tensor<4096xf32>
    return %result : tensor<4096xf32>
  }
}

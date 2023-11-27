builtin.module @module_a {
  func.func @abs(%input: tensor<4096xf32>) -> tensor<4096xf32> {
    %result = math.absf %input : tensor<4096xf32>
    return %result : tensor<4096xf32>
  }
}
